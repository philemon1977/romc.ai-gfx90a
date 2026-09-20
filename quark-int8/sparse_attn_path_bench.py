#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解码注意力微基准：量清 gfx90a 上"prefill-ragged 路径"与"decode 兜底路径"的真实代价与下限。

生产形状（由 backend 调用点与 profile 反推）：
  M=1（decode 单 token）、heads=128、head_dim=576（nope 512 + rope 64）、
  index_topk=2048、本地 KV = ctx/DCP = 8192 行（32K 上下文 / DCP=8 的上限档）。
对比：
  A) _rocm_sparse_attn_prefill_ragged_triton   ← 后端实际在用（grid = (M, cdiv(H,16))）
  B) _rocm_sparse_attn_decode_ragged_triton    ← gfx90a 上走"未调优兜底"（同尺寸网格、无 split-K）
并算出访存下限（每层只需读 TOPK x D x 2B）作为"该有多快"的尺子。
"""
import time
import torch
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _rocm_sparse_attn_prefill_ragged_triton,
    _rocm_sparse_attn_decode_ragged_triton,
)

DEV = "cuda"
H, D, NOPE, ROPE = 128, 576, 512, 64
NCTX, TOPK, BS = 8192, 2048, 16
SCALE = D ** -0.5


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def build(M):
    torch.manual_seed(0)
    q = (torch.randn(M, H, D, device=DEV, dtype=torch.bfloat16) * 0.1)
    cache = (torch.randn(NCTX, D, device=DEV, dtype=torch.bfloat16) * 0.1)
    if M == 1:
        idx = torch.randperm(NCTX, device=DEV)[:TOPK].to(torch.int32)
        indptr = torch.tensor([0, TOPK], device=DEV, dtype=torch.int32)
    else:
        rows = torch.stack([torch.randperm(NCTX, device=DEV)[:TOPK] for _ in range(M)])
        idx = rows.reshape(-1).to(torch.int32)
        indptr = torch.arange(0, (M + 1) * TOPK, TOPK, device=DEV, dtype=torch.int32)
    return q, cache, idx, indptr


def main():
    print("形状: M=1, heads=%d, head_dim=%d (nope %d + rope %d), topk=%d, cache=%d 行, page=%d"
          % (H, D, NOPE, ROPE, TOPK, NCTX, BS))
    floor_us = TOPK * D * 2 / 1.638e12 * 1e6   # 每层只需读 TOPK 行 KV（bf16）
    print("访存下限（每层读 %d x %d x 2B = %.2f MB @1638 GB/s）= %.1f us"
          % (TOPK, D, TOPK * D * 2 / 2**20, floor_us))
    print()
    for M in (1, 8):
        q, cache, idx, indptr = build(M)
        paged = cache.view(NCTX // BS, BS, D)
        lse = torch.empty(M, H, device=DEV, dtype=torch.float32)
        print("== M=%d ==" % M)
        try:
            outA = _rocm_sparse_attn_prefill_ragged_triton(
                q, cache, idx, indptr, SCALE, None, NOPE, ROPE, lse=lse)
            tA = bench(lambda: _rocm_sparse_attn_prefill_ragged_triton(
                q, cache, idx, indptr, SCALE, None, NOPE, ROPE, lse=lse))
            print("  A prefill-ragged  : %8.1f us/层   grid=(%d,%d)   ⇒ %.1f x 访存下限"
                  % (tA * 1e6, M, (H + 15) // 16, tA * 1e6 / floor_us))
        except Exception as exc:
            outA, tA = None, float("nan")
            print("  A prefill-ragged  : ERR %r" % (exc,))
        try:
            outB = _rocm_sparse_attn_decode_ragged_triton(
                q, paged, idx, indptr, SCALE, None, NOPE, ROPE)
            tB = bench(lambda: _rocm_sparse_attn_decode_ragged_triton(
                q, paged, idx, indptr, SCALE, None, NOPE, ROPE))
            print("  B decode-兜底     : %8.1f us/层   grid=(%d,%d)   ⇒ %.1f x 访存下限"
                  % (tB * 1e6, M, (H + 15) // 16, tB * 1e6 / floor_us))
        except Exception as exc:
            outB = None
            print("  B decode-兜底     : ERR %r" % (exc,))
        if outA is not None and outB is not None:
            num = (outA.float() - outB.float()).abs().mean().item()
            den = outB.float().abs().mean().item() + 1e-6
            print("  两路径数值差     : %.2f%%（同输入同 scale 下的相对误差）" % (100 * num / den))
        if outA is not None:
            print("  参考量级         : |out| 均值 %.4f, lse 均值 %.4f"
                  % (outA.float().abs().mean().item(), lse.float().mean().item()))
        print()


if __name__ == "__main__":
    main()