#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""集成形状对比：一次 launch（S*M 行，需重排索引）vs S 次 launch（M 行，索引原样）。
前者要一个小重排；后者零重排但吃 S 次启动开销。用 cudagraph 之外的裸 launch 量最坏情况。
"""
import time
import torch
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _rocm_sparse_attn_prefill_ragged_triton

DEV = "cuda"
H, D, NOPE, ROPE = 128, 576, 512, 64
SCALE = D ** -0.5


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
    torch.manual_seed(0)
    for M in (1, 4):
        nctx, topk, S = 1024, 1024, 8
        q = (torch.randn(M, H, D, device=DEV, dtype=torch.bfloat16) * 0.1)
        cache = (torch.randn(nctx, D, device=DEV, dtype=torch.bfloat16) * 0.1)
        idx2d = torch.stack([torch.randperm(nctx, device=DEV)[:topk] for _ in range(M)]).to(torch.int32)
        flat = idx2d.reshape(-1)
        base_ptr = torch.arange(0, (M + 1) * topk, topk, device=DEV, dtype=torch.int32)
        chunk = topk // S
        # (a) 一次 launch：把每查询的索引按 split 重排成 (q,s) 行序
        perm = (idx2d.view(M, S, chunk).permute(1, 0, 2).reshape(-1))
        indptr_a = torch.arange(0, (M * S + 1) * chunk, chunk, device=DEV, dtype=torch.int32)
        lse_a = torch.empty(M * S, H, device=DEV, dtype=torch.float32)
        q_a = q.repeat_interleave(S, dim=0).contiguous()
        # (b) S 次 launch：每次 M 行，索引原样，窗口是各查询的第 s 段
        indptr_bs, offs = [], []
        for s in range(S):
            ip = (base_ptr[:M] + s * chunk).to(torch.int32)
            ip = torch.cat([ip, (base_ptr[M] + 0).reshape(1)]).contiguous()
            indptr_bs.append(ip)
        # 注意：最后一次的末端要指向该查询第 s 段的结尾 ⇒ 用 base + (s+1)*chunk 逐查询构造
        indptr_bs = []
        for s in range(S):
            ip = torch.empty(M + 1, device=DEV, dtype=torch.int32)
            for i in range(M):
                ip[i] = i * topk + s * chunk
            ip[M] = (M - 1) * topk + (s + 1) * chunk
            indptr_bs.append(ip.contiguous())
        lse_b = torch.empty(M, H, device=DEV, dtype=torch.float32)
        print("== M=%d, S=%d ==" % (M, S))
        # 基准 S=1
        indptr1 = base_ptr.contiguous()
        lse1 = torch.empty(M, H, device=DEV, dtype=torch.float32)
        t1 = bench(lambda: _rocm_sparse_attn_prefill_ragged_triton(
            q, cache, flat, indptr1, SCALE, None, NOPE, ROPE, lse=lse1))
        print("  S=1 单次        : %8.1f us" % (t1 * 1e6))
        ta = bench(lambda: _rocm_sparse_attn_prefill_ragged_triton(
            q_a, cache, perm, indptr_a, SCALE, None, NOPE, ROPE, lse=lse_a))
        print("  (a) 一次 launch : %8.1f us  加速 %.2fx" % (ta * 1e6, t1 / ta))
        def run_b():
            for s in range(S):
                _rocm_sparse_attn_prefill_ragged_triton(
                    q, cache, flat, indptr_bs[s], SCALE, None, NOPE, ROPE, lse=lse_b)
        tb = bench(run_b)
        print("  (b) %d 次 launch  : %8.1f us  加速 %.2fx" % (S, tb * 1e6, t1 / tb))
        print()


if __name__ == "__main__":
    main()