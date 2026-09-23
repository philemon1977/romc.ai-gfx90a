#!/usr/bin/env python3
"""压缩 KV cache（fp8_ds_mla，**584 字节/行**）写→读往返。

为什么单独测：`kdtest_kvcache_decode_roundtrip` 用的是 **SWA cache 的 516 字节行**
（512 负载 + 4 scale），而 DSA/压缩路径走的是 **584 字节行**（512 负载 + 72 字节
"segregated UE8M0 scale" 区），写入函数是 `rope_quant_insert`（Triton），
与 SWA 的 C++ 编码器是**两条不同的路**。所以"SWA 往返通过"并不能覆盖它。

做法：恒等 RoPE ⇒ 写入退化为"fp8 量化 + 分页写入 + scale 布局"；
再用**已验证的读端** `rocm_sparse_attn_decode`（DSA 分支，单行选中）读回，
与输入 latent 比对。判据：nope/rope 两段都应与输入一致（fp8 精度内）。

用法（容器内，需 GPU）:
  python3 ktest_compressed_cache_roundtrip.py
"""
import torch

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
ROW = 584            # ← 压缩缓存行字节数（SWA 是 516）
BS = 32
NBLK = 4
T = 8

import vllm  # noqa: F401
from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import rope_quant_insert
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_decode

dev = "cuda"
torch.manual_seed(0)

# 恒等 RoPE（cos=1, sin=0）
cos_sin = torch.zeros(64, ROPE_DIM, dtype=torch.float32, device=dev)
cos_sin[:, : ROPE_DIM // 2] = 1.0

latent = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16, device=dev)
latent[:, :NOPE_DIM] = 0.5
latent[:, NOPE_DIM:] = 0.25
positions = torch.arange(T, dtype=torch.int64, device=dev)
slots = torch.arange(T, dtype=torch.int32, device=dev)          # 前 T 个 slot

cache3 = torch.zeros(NBLK, BS, ROW, dtype=torch.uint8, device=dev)
rope_quant_insert(latent, positions, cos_sin, cache3, slots, 2, fp8_scale=None)
torch.cuda.synchronize()

# ⚠️ "非空 slot" 不能用 any() 判：每行尾部有 scale 区，未写的行也可能非零。
#    正确判据来自 docstring："只写组边界 token" ⇒ ratio=2 时是**奇数位置**。
boundary = [int(s) for s in slots.tolist() if (int(s) % 2) == 1]
print(f"组边界 slot（ratio=2 ⇒ 奇数位置）: {boundary}；非边界 slot 读回应为 0（设计如此）")

ok = True
for s in boundary:
    out = torch.zeros(1, 8, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    q = torch.zeros(1, 8, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    q[:, :, :NOPE_DIM] = 0.01
    swa_idx = torch.full((1, 1, 128), -1, dtype=torch.int32, device=dev)
    swa_lens = torch.zeros(1, dtype=torch.int32, device=dev)
    topk_ragged = torch.tensor([s], dtype=torch.int32, device=dev)
    topk_indptr = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    rocm_sparse_attn_decode(
        q=q, kv_cache=cache3, swa_k_cache=cache3, swa_only=False,
        topk_indices=None, topk_lens=None,
        swa_indices=swa_idx, swa_lens=swa_lens,
        swa_ragged_indices=None, swa_ragged_indptr=None,
        topk_ragged_indices=topk_ragged, topk_ragged_indptr=topk_indptr,
        attn_sink=None, scale=1.0, head_dim=HEAD_DIM,
        nope_head_dim=NOPE_DIM, rope_head_dim=ROPE_DIM, output=out,
    )
    torch.cuda.synchronize()
    o = out[0, 0].float()
    nope, rope = o[:NOPE_DIM].mean().item(), o[NOPE_DIM:].mean().item()
    good = abs(nope - 0.5) < 0.03 and abs(rope - 0.25) < 0.03
    ok &= good
    print(f"  slot {s}: 读回 nope均值={nope:.4f}（输入 0.5） rope均值={rope:.4f}（输入 0.25） "
          f"-> {'PASS' if good else 'FAIL'}")

print("\n[PASS] 压缩缓存写(rope_quant_insert 584B) 与读端一致" if ok
      else "\n[FAIL] 压缩缓存写读不一致 —— 压缩 KV 路径有问题")
raise SystemExit(0 if ok else 1)
