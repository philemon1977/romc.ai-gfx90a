#!/usr/bin/env python3
"""压缩 KV cache 的**多行、逐行可区分**写→读往返。

旧版（ktest_compressed_cache_roundtrip.py）每行写的是**同样的值**(0.5/0.25)，
所以"行序错位 / 串行 scale / 部分写"这类 bug 它天然看不见；而 len(zip(scale,row))
式的错位在单行测试里也不会暴露。

本测试：每行给不同的值 val_t=(t+1)/32（nope 段）与 val_t/2（rope 段），
ratio=2 ⇒ 只有组边界（奇数位置）落地；再**逐行**读回并核对自己那一份值。

覆盖对象 `rope_quant_insert`（Triton）里有一处平台条件编译：
    _JOIN_ROW_PTRS = not current_platform.is_rocm()
即 ROCm 走的是另一条指针拼接路径——这类"只对非 CUDA 生效"的分支从没在 gfx90a 上验过，
正是 ⑫ mHC 那个 bug 的同一类风险面。
"""
import torch

HEAD_DIM, ROPE_DIM = 512, 64
NOPE = HEAD_DIM - ROPE_DIM
ROW, BS, NBLK = 584, 32, 8
T = 16
dev = "cuda"
torch.manual_seed(0)

from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import rope_quant_insert
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_decode

cos_sin = torch.zeros(64, ROPE_DIM, dtype=torch.float32, device=dev)
cos_sin[:, : ROPE_DIM // 2] = 1.0                      # 恒等旋转

vals = torch.arange(1, T + 1, dtype=torch.float32, device=dev) / 32.0
latent = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16, device=dev)
latent[:, :NOPE] = vals[:, None]
latent[:, NOPE:] = (vals / 2)[:, None]
positions = torch.arange(T, dtype=torch.int64, device=dev)
slots = torch.arange(T, dtype=torch.int32, device=dev)
cache3 = torch.zeros(NBLK, BS, ROW, dtype=torch.uint8, device=dev)

rope_quant_insert(latent, positions, cos_sin, cache3, slots, 2, fp8_scale=None)
torch.cuda.synchronize()

landed = [t for t in range(T) if t % 2 == 1]
print(f"逐行取值: nope=val_t, rope=val_t/2, val_t=(t+1)/32；ratio=2 ⇒ 落地行 {landed}")

def readback(t):
    out = torch.zeros(1, 8, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    q = torch.ones(1, 8, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    swa_idx = torch.full((1, 1, 128), -1, dtype=torch.int32, device=dev)
    swa_lens = torch.zeros(1, dtype=torch.int32, device=dev)
    rocm_sparse_attn_decode(
        q=q, kv_cache=cache3, swa_k_cache=cache3, swa_only=False,
        topk_indices=None, topk_lens=None, swa_indices=swa_idx, swa_lens=swa_lens,
        swa_ragged_indices=None, swa_ragged_indptr=None,
        topk_ragged_indices=torch.tensor([t], dtype=torch.int32, device=dev),
        topk_ragged_indptr=torch.tensor([0, 1], dtype=torch.int32, device=dev),
        attn_sink=None, scale=1.0, head_dim=HEAD_DIM,
        nope_head_dim=NOPE, rope_head_dim=ROPE_DIM, output=out)
    torch.cuda.synchronize()
    o = out[0, 0].float()
    return o[:NOPE].mean().item(), o[NOPE:].mean().item()

fails = 0
print(f"\n{'slot':>4} {'期望nope':>9} {'读回nope':>9} {'期望rope':>9} {'读回rope':>9}  判定")
for t in landed:
    want_n, want_r = float(vals[t]), float(vals[t]) / 2
    got_n, got_r = readback(t)
    ok = abs(got_n - want_n) < 0.03 and abs(got_r - want_r) < 0.03
    if not ok:
        fails += 1
    print(f"{t:>4} {want_n:>9.4f} {got_n:>9.4f} {want_r:>9.4f} {got_r:>9.4f}  {'PASS' if ok else '◆FAIL◆'}")

# 额外：把"读回值"当输入的函数做交叉检验——若行被错位，下面会看到读回 ≈ 别的行的值
print("\n交叉检验（读回值落在哪一行的输入上）：")
for t in landed[:4]:
    got_n, _ = readback(t)
    best = min(range(T), key=lambda u: abs(float(vals[u]) - got_n))
    print(f"  slot {t}: 读回 nope={got_n:.4f} ⇒ 最接近第 {best} 行的值 {float(vals[best]):.4f} "
          f"{'✅ 就是自己' if best == t else '◆错位◆'}")
    if best != t:
        fails += 1

print(f"\n结论：{'多行写读逐行一致 ✅' if fails == 0 else f'{fails} 处异常 ❌ —— 压缩 KV 多行路径有问题'}")
raise SystemExit(0 if fails == 0 else 1)
