#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""split-K 实验（不改内核）：把 (query, split) 当行喂给同一个 ragged 内核，再用 LSE 合并。

依据：内核写出口径 = out = acc/l_i、lse = m_i + log(l_i) ⇒ 合并式 w_s = softmax_s(lse_s)、
out = Σ_s w_s·out_s —— 与 DCP 跨 rank 合并同形。
目的：回答"gfx90a 上这个稀疏注意力内核能不能靠并行度救回来"。
"""
import time
import torch
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _rocm_sparse_attn_prefill_ragged_triton

DEV = "cuda"
H, D, NOPE, ROPE = 128, 576, 512, 64
NCTX, TOPK = 8192, 2048
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
    q1 = (torch.randn(1, H, D, device=DEV, dtype=torch.bfloat16) * 0.1)
    cache = (torch.randn(NCTX, D, device=DEV, dtype=torch.bfloat16) * 0.1)
    rows = torch.randperm(NCTX, device=DEV)[:TOPK].to(torch.int32)          # 单查询选中的 2048 行
    ref_out, ref_lse = None, None
    print("M=1, topk=%d, cache=%d 行, 访存下限 %.1f us/层" % (TOPK, NCTX, TOPK * D * 2 / 1.638e12 * 1e6))
    print("%-4s %10s %10s %10s %8s %s" % ("S", "内核 us", "合并 us", "合计 us", "加速", "对拍"))
    for S in (1, 2, 4, 8, 16):
        per = (TOPK + S - 1) // S
        chunks = [rows[i * per:(i + 1) * per] for i in range(S)]
        chunks = [c for c in chunks if c.numel() > 0]
        idx = torch.cat(chunks)
        lens = torch.tensor([c.numel() for c in chunks], device=DEV, dtype=torch.int32)
        indptr = torch.zeros(S + 1, device=DEV, dtype=torch.int32)
        indptr[1:] = torch.cumsum(lens, 0)
        q_rep = q1.repeat_interleave(S, dim=0).contiguous()
        out_s = torch.empty(S, H, D, device=DEV, dtype=torch.bfloat16)
        lse_s = torch.empty(S, H, device=DEV, dtype=torch.float32)

        def run_kernel():
            _rocm_sparse_attn_prefill_ragged_triton(
                q_rep, cache, idx, indptr, SCALE, None, NOPE, ROPE, lse=lse_s)
        t_k = bench(run_kernel)
        run_kernel()

        def run_combine():
            w = torch.softmax(lse_s.float(), dim=0)                 # [S, H]
            torch.sum(out_s.float() * w[:, :, None], dim=0, out=out_c)
        out_c = torch.empty(H, D, device=DEV, dtype=torch.float32)
        t_c = bench(run_combine)
        run_combine()
        if S == 1:
            ref_out, ref_lse = out_c.clone(), lse_s.clone()
            err = "—"
        else:
            num = (out_c - ref_out).abs().mean().item()
            den = ref_out.abs().mean().item() + 1e-6
            err = "%.3f%%" % (100 * num / den)
        print("%-4d %10.1f %10.1f %10.1f %8s %s" % (S, t_k * 1e6, t_c * 1e6, (t_k + t_c) * 1e6,
              ("%.2fx" % (t1 / (t_k + t_c))) if "t1" in dir() else "—", err))
        if S == 1:
            t1 = t_k + t_c
    print()
    print("注：合并目前是 4 个 torch 算子；若需可写成单内核，但先看量级。")


if __name__ == "__main__":
    main()