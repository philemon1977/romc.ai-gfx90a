#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补齐两种 gemm2 分片假设 + 大 pairs 验证：
  A) gemm2 切 N：K=2048, N=768   （若 TP 切 down_proj 的输出维）
  B) gemm2 切 K：K=256,  N=6144  （若 TP 切 down_proj 的输入维——vLLM FusedMoE 的常规做法）
  C) gemm1 pairs=256（M=32）验证 BN=16 在大批下是否仍最优
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_gemv_bench import bench                 # noqa: E402
from moe_gemv_cmp import build, ref_out, rel_err  # noqa: E402

torch.manual_seed(0)
B1, S1, B2, S2 = build(6, 8)
sys.path.insert(0, "/patches/moe_gemv")
import mi250_moe_gemv_gs as M
import mi250_moe_gemv_v3 as V3

CASES = [
    ("gemm2-A 切N  K=2048 N=768",  B2[:, :768].contiguous(),             S2[:, :768].contiguous(),             "K2048N768"),
    ("gemm2-B 切K  K=256  N=6144", B2[:, :, :128].contiguous(),          S2[:, :, :4].contiguous(),            "K256N6144"),
]


def run_case(name, B, S, tag, toks, topk=8):
    E, N, Kh = B.shape
    K = Kh * 2
    gs = K // S.shape[2]
    nbytes = B.numel() + S.numel() * 2
    print()
    print("== %s : K=%d N=%d gs=%d 每 token %.2f MB ==" % (name, K, N, gs, nbytes / 2**20))
    for M_ in toks:
        pairs = M_ * topk
        X = torch.randn(M_, K, device="cuda", dtype=torch.bfloat16)
        ids = torch.arange(pairs, device="cuda", dtype=torch.int32) % E
        wts = torch.rand(pairs, device="cuda", dtype=torch.float32) * 0.5 + 0.25
        O = torch.empty(pairs, N, device="cuda", dtype=torch.bfloat16)
        rows = []
        for bn in (16, 32, 64):
            if N % bn:
                continue
            for g in (4, 8):
                if g * gs > K or K % (g * gs):
                    continue
                for nw in (1, 2, 4):
                    grid = (pairs, N // bn)
                    try:
                        def run():
                            V3._gemv_moe_v3[grid](X, B, S, O, ids, wts, M_, K, N, topk,
                                                  APPLY_W=True, GROUP=gs, G_PER_STEP=g,
                                                  BLOCK_N=bn, num_warps=nw)
                        with torch.no_grad():
                            dt = bench(run, iters=20 if pairs <= 64 else 8, warmup=3)
                    except Exception as exc:
                        continue
                    rows.append((dt, bn, g, nw))
        rows.sort()
        for dt, bn, g, nw in rows[:6]:
            print("    M=%-3d pairs=%-3d BN=%-3d G=%d w=%d %9.1f us  %7.1f GB/s" % (
                M_, pairs, bn, g, nw, dt * 1e6, nbytes / dt / 1e9))
        if rows:
            print("    => 最优 BN=%d G=%d w=%d (%.1f us)" % (rows[0][1], rows[0][2], rows[0][3], rows[0][0] * 1e6))
            bn, bk, nw = 64, 64, 4
            grid = (pairs, N // bn)
            def run1():
                M._gemv_moe_k[grid](X, B, S, O, ids, wts, M_, K, N, topk, APPLY_W=True,
                                    GROUP=gs, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw)
            dt = bench(run1, iters=20 if pairs <= 64 else 8, warmup=3)
            print("    (v1 BN=64 BK=64 w=4: %.1f us  %.1f GB/s)" % (dt * 1e6, nbytes / dt / 1e9))


for name, B, S, tag in CASES:
    run_case(name, B, S, tag, [1])
print()
run_case("gemm1 大 pairs 验证", B1[:, :512].contiguous(), S1[:, :512].contiguous(), "K6144N512", [8, 16, 32])
