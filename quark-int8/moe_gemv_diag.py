#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定位实验：60 GB/s 的天花板到底是谁。

已证伪：跨 lane 归约（v2 二维累加器 1.01x，无提速）、纯 ALU 上限（0.9 Tops/s = 4% 峰值）。
待判：访存模式（每行 32-64 B 连续、跨行 stride 3072 B）是否为瓶颈。
A. 纯 load（同 v2 模式，只累加原始字节，几乎无 ALU）
B. 纯 load（一程序一行：整行 3072 B 连续流）
C. 纯 load（程序读连续大块，对照理论上限）
D. torch.sum(108 MB) 对照（PyTorch 能到多少）
E. v2 各配置的 n_regs / n_spills
"""
import argparse, os, sys, time
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@triton.jit
def _load_strided(W, O, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_p = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k2 = tl.arange(0, BLOCK_K // 2)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        wb = tl.load(W + pid_p * N * (K // 2) + offs_n[:, None] * (K // 2) + (k0 // 2 + offs_k2)[None, :])
        acc += tl.sum(wb.to(tl.float32), axis=1)
    tl.store(O + pid_p * N + offs_n, acc)


@triton.jit
def _load_row(W, O, K, BLOCK: tl.constexpr):
    """一程序一行：整行 K/2 字节连续流。"""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k0 in range(0, K // 2, BLOCK):
        wb = tl.load(W + pid * (K // 2) + k0 + offs)
        acc += wb.to(tl.float32)
    tl.store(O + pid, tl.sum(acc, axis=0))


@triton.jit
def _load_contig(W, O, BLOCK: tl.constexpr):
    """程序读连续大块（对照上限）。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(W + offs)
    tl.store(O + pid, tl.sum(v.to(tl.float32), axis=0))


@triton.jit
def _deq_nowb_raw(X, W, S, O, IDS, WTS, M, K, N, TOPK, GROUP: tl.constexpr,
                  APPLY_W: tl.constexpr, MODE: tl.constexpr,
                  BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """v2 内核的消融版：MODE 0=全量 1=去掉 scale 2=去掉 nibble 提取。"""
    pid_p = tl.program_id(0)
    pid_n = tl.program_id(1)
    t = pid_p // TOPK
    e = tl.maximum(tl.load(IDS + pid_p), 0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k2 = tl.arange(0, BLOCK_K // 2)
    acc2 = tl.zeros((BLOCK_N, BLOCK_K // 2), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        wb = tl.load(W + e * N * (K // 2) + offs_n[:, None] * (K // 2) + (k0 // 2 + offs_k2)[None, :])
        wi = wb.to(tl.int32)
        for j in tl.static_range(2):
            kk = k0 + offs_k2 * 2 + j
            if MODE == 2:
                val = wb.to(tl.float32)
            else:
                val = (((wi >> (4 * j)) & 0xF) - 8).to(tl.float32)
            xj = tl.load(X + t * K + kk).to(tl.float32)
            if MODE == 1:
                acc2 += val * xj[None, :]
            else:
                sc = tl.load(S + e * N * (K // GROUP) + offs_n[:, None] * (K // GROUP) + (kk // GROUP)[None, :])
                acc2 += val * xj[None, :] * sc.to(tl.float32)
    acc = tl.sum(acc2, axis=1)
    if APPLY_W:
        acc = acc * tl.load(WTS + pid_p)
    tl.store(O + pid_p * N + offs_n, acc.to(O.dtype.element_ty))


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--tokens", default="1,8")
    ap.add_argument("--topk", type=int, default=8)
    a = ap.parse_args()
    from moe_gemv_cmp import build
    B1, S1, B2, S2 = build(6, a.experts)
    B = B1.contiguous()
    E, N, Kh = B.shape
    K = Kh * 2
    group = K // S1.shape[2]
    nbytes = B.numel()
    print("B1 %s  K=%d N=%d gs=%d  %.1f MB" % (tuple(B.shape), K, N, group, nbytes / 2**20))
    nb = B.view(-1)

    print()
    print("== A. 纯 load：v2 同模式（每行 32B 连续 / 跨行 stride 3072B）==")
    for M in [int(x) for x in a.tokens.split(",")]:
        pairs = M * a.topk
        O = torch.empty(pairs, N, device="cuda", dtype=torch.float32)
        for bn, bk, nw in [(64, 64, 4), (128, 128, 4), (64, 256, 4)]:
            if N % bn:
                continue
            grid = (pairs, N // bn)
            dt = bench(lambda: _load_strided[grid](B, O, N, K, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw))
            tot = nbytes * pairs / E
            print("   M=%-3d(pairs %-3d) BN=%-4d BK=%-4d w=%d  %9.1f us  %7.1f GB/s" % (
                M, pairs, bn, bk, nw, dt * 1e6, tot / dt / 1e9))

    print()
    print("== B. 纯 load：一程序一行（整行连续）==")
    for M in [int(x) for x in a.tokens.split(",")]:
        pairs = M * a.topk
        O = torch.empty(pairs * N, device="cuda", dtype=torch.float32)
        assert (K // 2) % 1024 == 0, "BLOCK 必须整除 K//2"
        for blk, nw in [(256, 4), (512, 4), (1024, 4)]:
            grid = (pairs * N,)
            dt = bench(lambda: _load_row[grid](B, O, K, BLOCK=blk, num_warps=nw))
            tot = nbytes * pairs / E
            print("   M=%-3d(pairs %-3d) BLOCK=%-5d w=%d  %9.1f us  %7.1f GB/s" % (
                M, pairs, blk, nw, dt * 1e6, tot / dt / 1e9))

    print()
    print("== C. 纯 load：连续大块（对照上限）==")
    per = nbytes // 512                      # 每个程序读 nbytes/512 字节，整除
    for blk, nprog in [(4096, per // 4096), (16384, per // 16384)]:
        O = torch.empty(nprog, device="cuda", dtype=torch.float32)
        cov = blk * nprog
        dt = bench(lambda: _load_contig[(nprog,)](nb, O, BLOCK=blk, num_warps=4))
        print("   BLOCK=%-6d progs=%-5d 覆盖 %5.1f MB  %9.1f us  %7.1f GB/s" % (
            blk, nprog, cov / 2**20, dt * 1e6, cov / dt / 1e9))

    print()
    print("== D. torch 对照 ==")
    for tag, fn in (("B1.sum(int32)", lambda: B.to(torch.int32).sum()),
                    ("B1.float().sum()", lambda: B.float().sum())):
        dt = bench(fn, iters=10, warmup=2)
        print("   %-18s %9.1f us  %7.1f GB/s" % (tag, dt * 1e6, nbytes / dt / 1e9))

    print()
    print("== E. 消融：v2 内核裁掉 scale / nibble 提取 ==")
    X = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(a.topk, device="cuda", dtype=torch.int32)
    wts = torch.ones(a.topk, device="cuda", dtype=torch.float32)
    O = torch.empty(a.topk, N, device="cuda", dtype=torch.bfloat16)
    names = {0: "全量(v2)", 1: "去 scale", 2: "去 nibble(仅 wb*x)"}
    for mode in (0, 1, 2):
        for bn, bk, nw in [(64, 64, 4), (128, 128, 4)]:
            grid = (a.topk, N // bn)
            kern = _deq_nowb_raw[grid](X, B, S1, O, ids, wts, 1, K, N, a.topk, GROUP=group,
                                       APPLY_W=True, MODE=mode, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw)
            dt = bench(lambda: _deq_nowb_raw[grid](X, B, S1, O, ids, wts, 1, K, N, a.topk, GROUP=group,
                                                   APPLY_W=True, MODE=mode, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw))
            print("   %-16s BN=%-4d BK=%-4d  %9.1f us  %7.1f GB/s  regs=%s spills=%s" % (
                names[mode], bn, bk, dt * 1e6, nbytes / dt / 1e9,
                getattr(kern, "n_regs", "?"), getattr(kern, "n_spills", "?")))


if __name__ == "__main__":
    main()
