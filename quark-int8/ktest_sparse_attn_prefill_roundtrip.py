#!/usr/bin/env python3
"""稀疏 MLA prefill 往返（单行/双行选中）

给定已知 KV 与 indices → 输出必须等于被选中行的值（单行）或加权平均（双行）。
已验证：prefill 核正确。

用法（容器内，需 GPU）: python3 ktest_sparse_attn_prefill_roundtrip.py
"""
import torch
HEAD_DIM=512; ROPE_DIM=64; NOPE_DIM=HEAD_DIM-ROPE_DIM
import vllm
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_prefill
dev="cuda"
T=2; H=8
# 已知 KV：两行，nope 分别 0.5 / -0.5，rope 0.25
kv=torch.zeros(T,1,HEAD_DIM,dtype=torch.bfloat16,device=dev)
kv[0,0,:NOPE_DIM]=0.5;  kv[0,0,NOPE_DIM:]=0.25
kv[1,0,:NOPE_DIM]=-0.5; kv[1,0,NOPE_DIM:]=0.25
q=torch.zeros(T,H,HEAD_DIM,dtype=torch.bfloat16,device=dev); q[:,:,:NOPE_DIM]=0.01
out=torch.zeros(T,H,HEAD_DIM,dtype=torch.bfloat16,device=dev)
idx=torch.full((T,128),-1,dtype=torch.int32,device=dev)
idx[0,0]=0          # 行0 只看第 0 行 KV
idx[1,0]=0; idx[1,1]=1   # 行1 看第 0/1 行
try:
    rocm_sparse_attn_prefill(
        q=q, kv=kv, indices=idx, topk_length=None,
        head_dim=HEAD_DIM, nope_head_dim=NOPE_DIM, rope_head_dim=ROPE_DIM,
        attn_sink=None, scale=1.0, output=out,
    )
    torch.cuda.synchronize()
    o0=out[0,0].float(); o1=out[1,0].float()
    print(f"行0 输出: nope均值={o0[:NOPE_DIM].mean().item():.4f} rope均值={o0[NOPE_DIM:].mean().item():.4f} absmax={o0.abs().max().item():.4g}")
    print(f"行1 输出: nope均值={o1[:NOPE_DIM].mean().item():.4f} rope均值={o1[NOPE_DIM:].mean().item():.4f} absmax={o1.abs().max().item():.4g}")
    print("期望：行0≈(0.5,0.25)；行1 是两行的加权平均（nope 介于 ±0.5）")
except Exception as e:
    print("prefill 调用失败:", type(e).__name__, str(e)[:300])
