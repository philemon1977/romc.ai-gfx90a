#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""split-K 集成测试：走 rocm_sparse_attn_prefill（我的分支）vs 低层内核 S=1 逐位对拍。

关键点：查询优先行序 r = i*S+s 是否与源 ragged_indices 对齐（M>1 才会暴露）。
"""
import torch
import vllm.v1.attention.ops.rocm_aiter_mla_sparse as M

DEV = "cuda"
H, D, NOPE, ROPE = 128, 576, 512, 64
SCALE = D ** -0.5


def case(Mq, nctx, topk, S):
    torch.manual_seed(1234)
    q = (torch.randn(Mq, H, D, device=DEV, dtype=torch.bfloat16) * 0.1)
    cache = (torch.randn(nctx, D, device=DEV, dtype=torch.bfloat16) * 0.1)
    rows = torch.stack([torch.randperm(nctx, device=DEV)[:topk] for _ in range(Mq)]).to(torch.int32)
    flat = rows.reshape(-1).contiguous()
    indptr = torch.arange(0, (Mq + 1) * topk, topk, device=DEV, dtype=torch.int32)
    kv3 = cache.view(nctx, 1, D)
    # 参考：低层内核，S=1
    ref_out = torch.empty(Mq, H, D, device=DEV, dtype=torch.bfloat16)
    ref_lse = torch.empty(Mq, H, device=DEV, dtype=torch.float32)
    M._rocm_sparse_attn_prefill_ragged_triton(
        q=q, kv=cache, indices=flat, indptr=indptr, scale=SCALE, attn_sink=None,
        nope_head_dim=NOPE, rope_head_dim=ROPE, lse=ref_lse).clone()
    ref_out = ref_out  # 占位（下面用返回值）
    ref = M._rocm_sparse_attn_prefill_ragged_triton(
        q=q, kv=cache, indices=flat, indptr=indptr, scale=SCALE, attn_sink=None,
        nope_head_dim=NOPE, rope_head_dim=ROPE, lse=ref_lse)
    # 待测：入口函数（分支），S 由模块级门控控制
    M._SPARSE_SPLITK = S
    out = torch.zeros(Mq, H, D, device=DEV, dtype=torch.bfloat16)
    lse = torch.zeros(Mq, H, device=DEV, dtype=torch.float32)
    M.rocm_sparse_attn_prefill(
        q=q, kv=kv3, indices=flat, topk_length=None, scale=SCALE, head_dim=D,
        nope_head_dim=NOPE, rope_head_dim=ROPE, attn_sink=None, output=out,
        ragged_indices=flat, ragged_indptr=indptr, output_lse=lse)
    d_out = (out.float() - ref.float()).abs().max().item()
    d_lse = (lse - ref_lse).abs().max().item()
    rel = (out.float() - ref.float()).abs().mean().item() / (ref.float().abs().mean().item() + 1e-9)
    print("  M=%-2d S=%-2d  max|Δout|=%.3e  max|Δlse|=%.3e  相对误差=%.3e  %s" % (
        Mq, S, d_out, d_lse, rel, "PASS" if (d_out == 0.0 and d_lse == 0.0) else ("近似" if rel < 1e-3 else "FAIL")))
    return d_out == 0.0 and d_lse == 0.0


def main():
    print("集成测试：入口函数（split-K 分支） vs 低层内核 S=1")
    ok = True
    for Mq in (1, 4, 8):
        for S in (2, 4, 8):
            ok &= case(Mq, 1024, 1024, S)
    print("全部逐位一致" if ok else "存在差异（见上）")


if __name__ == "__main__":
    main()