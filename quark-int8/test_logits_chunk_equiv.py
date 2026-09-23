# -*- coding: utf-8 -*-
"""等价性验证：分块后的 fp8_mqa_logits_torch == 上游一次性物化版本（逐元素）。

在容器里跑（CPU 张量），挂载的是**真正要上线的那个补丁文件**。
"""
import os, sys, torch

import vllm.v1.attention.ops.rocm_aiter_mla_sparse as M

def ref_impl(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
    """上游原版（分块前）—— 逐字照抄，作为唯一判据。"""
    k_fp8, scale = kv
    seq_len_kv = k_fp8.shape[0]
    k = k_fp8.to(torch.bfloat16)
    q = q.to(torch.bfloat16)
    device = q.device
    mask_lo = torch.arange(0, seq_len_kv, device=device)[None, :] >= cu_seqlen_ks[:, None]
    mask_hi = torch.arange(0, seq_len_kv, device=device)[None, :] < cu_seqlen_ke[:, None]
    mask = mask_lo & mask_hi
    score = torch.einsum("mhd,nd->hmn", q, k).float() * scale.reshape(-1)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))
    return logits

def build(M_, H, D, N, causal=True, seed=0, use_fp8=True):
    g = torch.Generator().manual_seed(seed)
    qf = torch.randn(M_, H, D, generator=g) * 0.5
    kf = torch.randn(N, D, generator=g) * 0.5
    if use_fp8:
        try:
            q = qf.to(torch.float8_e4m3fn); k = kf.to(torch.float8_e4m3fn)
            _ = k.to(torch.bfloat16)   # 确认 CPU 上 fp8->bf16 可用
        except Exception as e:
            print("  [warn] fp8 不可用(%s)，退回 bf16 输入" % type(e).__name__)
            q, k = qf.to(torch.bfloat16), kf.to(torch.bfloat16)
    else:
        q, k = qf.to(torch.bfloat16), kf.to(torch.bfloat16)
    scale = (torch.rand(N, 1, generator=g) * 0.1 + 0.01).to(torch.float32)
    weights = torch.rand(M_, H, generator=g).to(torch.float32)
    if causal:
        ke = torch.arange(1, M_ + 1, dtype=torch.int32).clamp(max=N)
        ks = torch.zeros(M_, dtype=torch.int32)
    else:
        ks = torch.zeros(M_, dtype=torch.int32)
        ke = torch.full((M_,), N, dtype=torch.int32)
    return q, (k, scale), weights, ks, ke

def cmp(name, a, b, tol=0.0):
    both_inf = torch.isinf(a) & torch.isinf(b) & (a.sign() == b.sign())
    finite = ~torch.isinf(a) & ~torch.isinf(b)
    ok_shape = a.shape == b.shape
    bad_finite = (torch.isinf(a) != torch.isinf(b))
    diff = 0.0
    if finite.any():
        diff = float((a[finite] - b[finite]).abs().max())
    exact = bool((a[both_inf | finite] == b[both_inf | finite]).all()) if ok_shape else False
    print("  %-34s shape=%s  finite_max_abs_diff=%.3e  inf_pattern_match=%s  bitwise_equal=%s"
          % (name, tuple(a.shape), diff, not bool(bad_finite.any()), exact))
    return ok_shape and not bool(bad_finite.any()) and exact

print("env VLLM_SPARSE_INDEXER_MAX_LOGITS_MB =", os.environ.get("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"))
print("env MI250_INDEXER_LOGITS_DEBUG       =", os.environ.get("MI250_INDEXER_LOGITS_DEBUG"))
print("module file:", M.__file__)
allok = True
cases = [
    ("causal M=64 H=4 N=256",   dict(M_=64,  H=4,  D=128, N=256,  causal=True)),
    ("full   M=64 H=4 N=256",   dict(M_=64,  H=4,  D=128, N=256,  causal=False)),
    ("causal M=128 H=8 N=512",  dict(M_=128, H=8,  D=128, N=512,  causal=True)),
    ("full   M=1  H=4 N=64",    dict(M_=1,   H=4,  D=128, N=64,   causal=False)),
    ("causal M=33 H=3 N=100",   dict(M_=33,  H=3,  D=128, N=100,  causal=True)),
    # ★ 强制重度分块（budget_mb=1 时 m_chunk=4 ⇒ 64 块），验证逐块 mask 正确
    ("HEAVY causal M=256 H=64 N=1024", dict(M_=256, H=64, D=128, N=1024, causal=True)),
    ("HEAVY full   M=200 H=32 N=1024", dict(M_=200, H=32, D=128, N=1024, causal=False)),
    ("HEAVY causal M=129 H=48 N=768",  dict(M_=129, H=48, D=128, N=768,  causal=True)),
]
for name, kw in cases:
    q, kv, w, ks, ke = build(**kw)
    a = ref_impl(q, kv, w, ks, ke)
    b = M.fp8_mqa_logits_torch(q, kv, w, ks, ke)
    allok &= cmp(name, a, b)

print("\nRESULT:", "PASS 全部分块等价" if allok else "FAIL 存在不一致")
sys.exit(0 if allok else 1)
