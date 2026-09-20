#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""split-K 的生产条件复测 + nan 归因。
生产条件（DCP=8）：本地 KV 只有 ctx/8 ≈ 1024 行；index_topk=2048 > 本地行数 ⇒ 选中集合里
有重复/覆盖全部本地行；索引顺序按 score 而非按位置（这里用随机顺序近似最坏、用排序近似最好）。
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


def run_case(name, nctx, topk, sorted_idx):
    torch.manual_seed(0)
    q1 = (torch.randn(1, H, D, device=DEV, dtype=torch.bfloat16) * 0.1)
    cache = (torch.randn(nctx, D, device=DEV, dtype=torch.bfloat16) * 0.1)
    if topk <= nctx:
        rows = torch.randperm(nctx, device=DEV)[:topk]
    else:
        rows = torch.randint(0, nctx, (topk,), device=DEV)   # 有重复（生产里 index_topk > 本地行数时）
    rows = rows.sort().values if sorted_idx else rows
    rows = rows.to(torch.int32)
    print("-- %s (local=%d, topk=%d, %s索引)" % (name, nctx, topk, "排序" if sorted_idx else "随机"))
    ref = None
    t1 = None
    for S in (1, 4, 8):
        per = (int(rows.numel()) + S - 1) // S
        chunks = [rows[i * per:(i + 1) * per] for i in range(S)]
        chunks = [c for c in chunks if c.numel() > 0]
        idx = torch.cat(chunks)
        lens = torch.tensor([c.numel() for c in chunks], device=DEV, dtype=torch.int32)
        indptr = torch.zeros(len(chunks) + 1, device=DEV, dtype=torch.int32)
        indptr[1:] = torch.cumsum(lens, 0)
        q_rep = q1.repeat_interleave(len(chunks), dim=0).contiguous()
        out_s = torch.empty(len(chunks), H, D, device=DEV, dtype=torch.bfloat16)
        lse_s = torch.empty(len(chunks), H, device=DEV, dtype=torch.float32)

        def run():
            _rocm_sparse_attn_prefill_ragged_triton(
                q_rep, cache, idx, indptr, SCALE, None, NOPE, ROPE, lse=lse_s)
        t = bench(run)
        run()
        w = torch.softmax(lse_s.float(), dim=0)
        out_c = torch.sum(out_s.float() * w[:, :, None], dim=0)
        msg = ""
        if S == 1:
            ref = out_c.clone()
            t1 = t
        else:
            d = (out_c - ref).abs()
            msg = "对拍 max|Δ|=%.2e mean|ref|=%.4f nan_out=%d nan_lse=%d" % (
                d.max().item(), ref.abs().mean().item(),
                int(torch.isnan(out_c).sum()), int(torch.isnan(lse_s).sum()))
        print("   S=%-3d %8.1f us  %s%s" % (S, t * 1e6,
              ("加速 %.2fx  " % (t1 / t)) if t1 else "", msg))
    print()


def main():
    run_case("生产近似：本地 1024 行 / 选 1024 / 随机序", 1024, 1024, False)
    run_case("生产近似：本地 1024 行 / 选 1024 / 排序", 1024, 1024, True)
    run_case("早期档：本地 4096 行 / 选 2048", 4096, 2048, False)


if __name__ == "__main__":
    main()