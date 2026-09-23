# -*- coding: utf-8 -*-
"""gfx90a 稀疏 indexer：aiter 真内核 vs torch 回退 —— 数值对拍 + 图捕获测试。

背景：vLLM rocm_aiter_ops.is_enabled() 被 @if_aiter_supported 包着，后者要求
get_cdna_version() > 2（CDNA3+）⇒ gfx90a 恒 None ⇒ indexer 一直走
fp8_paged_mqa_logits_torch（纯 Python 循环 + .item()，慢且不可进图）。
本脚本验证"按模块可用即用"接管后，aiter 的 Triton 内核在 gfx90a 上：
  (1) 数值与参考实现一致；
  (2) 能在 HIP graph 捕获内执行（这是 FULL cudagraph 的前提）。
"""
import os
import torch

os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")   # 刻意不开全局开关，验证"模块可用即用"
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (  # noqa: E402
    rocm_fp8_paged_mqa_logits,
    fp8_paged_mqa_logits_torch,
)
from vllm.v1.worker.workspace import init_workspace_manager  # noqa: E402

init_workspace_manager(torch.device("cuda"))   # aiter 路径要用 workspace

torch.manual_seed(0)
D, H, BS = 128, 32, 64   # 真实维度：index_head_dim=128, index_n_heads=32, kv block=64
NB, B, CTX, MML = 16, 2, [128, 200], 256
dev = "cuda"

k = (torch.randn(NB * BS, D, device=dev) * 0.15)
sc = (torch.rand(NB * BS, device=dev) * 0.5 + 0.75)
# 真实页内布局（权威来源：indexer_k_quant_and_cache_triton 写入端）：
#   每页 = [block_size × head_dim fp8 值][block_size × 4B fp32 scale]
kv = torch.zeros(NB, BS * D + BS * 4, dtype=torch.uint8, device=dev)
kv[:, : BS * D] = k.to(torch.float8_e4m3fn).view(torch.uint8).view(NB, BS * D)
kv[:, BS * D :] = sc.view(torch.float32).view(torch.uint8).view(NB, BS * 4)
kv = kv.view(NB, BS, 1, D + 4)

q = (torch.randn(B, 1, H, D, device=dev) * 0.15).to(torch.float8_e4m3fn)
w = torch.rand(B, H, device=dev)
ctx = torch.tensor(CTX, dtype=torch.int32, device=dev)
bt = torch.arange(NB, dtype=torch.int32, device=dev).view(1, -1).repeat(B, 1)

print("=== 1) 数值对拍（torch 回退 = 参考） ===")
ref = fp8_paged_mqa_logits_torch(q, kv, w, ctx, bt, MML)
out = rocm_fp8_paged_mqa_logits(q, kv, w, ctx, bt, None, MML)
print("  ref", tuple(ref.shape), "out", tuple(out.shape))
fin = torch.isfinite(ref) & torch.isfinite(out)
m = fin & torch.isfinite(ref)
ok_mask = torch.equal(torch.isfinite(ref), torch.isfinite(out))
print("  -inf 掩码一致:", ok_mask)
if m.any():
    a, b = out[m].float(), ref[m].float()
    rel = ((a - b).norm() / b.norm()).item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    print("  有限元数 %d  最大绝对差 %.3e  相对L2 %.3e  cos %.6f" % (int(m.sum()), (a - b).abs().max().item(), rel, cos))

print("=== 2) 图捕获测试（FULL cudagraph 的前提） ===")
try:
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            rocm_fp8_paged_mqa_logits(q, kv, w, ctx, bt, None, MML)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        cap = rocm_fp8_paged_mqa_logits(q, kv, w, ctx, bt, None, MML)
    g.replay()
    torch.cuda.synchronize()
    same = torch.equal(torch.isfinite(cap), torch.isfinite(out))
    d = (cap[torch.isfinite(cap)] - out[torch.isfinite(out)]).abs().max().item()
    print("  ✅ 捕获成功；replay 与即时执行一致: mask=%s 最大差 %.3e" % (same, d))
except Exception as e:
    print("  ❌ 捕获失败:", type(e).__name__, str(e)[:220])