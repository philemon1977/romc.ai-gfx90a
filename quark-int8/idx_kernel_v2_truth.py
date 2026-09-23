# -*- coding: utf-8 -*-
"""以"原始 k"为真值的三方对拍，定位 kernel 与参考的 cos 0.915 差在哪。"""
import os, torch
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    rocm_fp8_paged_mqa_logits, fp8_paged_mqa_logits_torch, indexer_k_quant_and_cache_triton)
from vllm.v1.worker.workspace import init_workspace_manager
init_workspace_manager(torch.device("cuda"))
torch.manual_seed(0)
D, BS, NB, H, B = 128, 64, 8, 32, 2
NT = NB * BS
MML = 512
dev = "cuda"
kv = torch.zeros(NB, BS, D + 4, dtype=torch.uint8, device=dev)
k = (torch.randn(NT, D, device=dev) * 0.3).to(torch.float8_e4m3fn)
slot = torch.arange(NT, dtype=torch.int64, device=dev)
indexer_k_quant_and_cache_triton(k, kv, slot, 128, None)
kv4 = kv.view(NB, BS, 1, D + 4)
kvf = kv.view(NB, -1)
vals = kvf[:, : BS * D].view(torch.float8_e4m3fn).float()
scales = kvf[:, BS * D :].contiguous().view(torch.float32)
tt = torch.arange(BS, device=dev); dd = torch.arange(D, device=dev)
shuf = ((tt[:, None] // 16) * (16 * D) + (tt[:, None] % 16) * 16 + (dd[None, :] // 16) * 256 + (dd[None, :] % 16)).reshape(-1)
kd = vals[:, shuf].view(NB, BS, D)                # 反解出的 k（应与写入端一致）
q = (torch.randn(B, 1, H, D, device=dev) * 0.3).to(torch.float8_e4m3fn)
w = torch.rand(B, H, device=dev)
ctx = torch.tensor([256, 480], dtype=torch.int32, device=dev)
bt = torch.arange(NB, dtype=torch.int32, device=dev).view(1, -1).repeat(B, 1)

def logits_from(dense_k):
    """用给定 dense k（[NT,D] 或 [NB,BS,D]）+ 缓存里的 scale 复算 logits。"""
    kk = dense_k.reshape(NB, BS, D)
    out = torch.full((B, MML), float("-inf"), device=dev)
    for b in range(B):
        c = int(ctx[b]); np_ = (c + BS - 1) // BS
        kcat = torch.cat([kk[bt[b, p]] for p in range(np_)])[:c]
        scat = torch.cat([scales[bt[b, p]] for p in range(np_)])[:c]
        lg = (torch.relu(q.float()[b, 0] @ kcat.T) * w[b][:, None]).sum(0) * scat
        out[b, :c] = lg
    return out

ref_orig = logits_from(k.float())      # 真值：原始 k（未经过缓存）
ref_deq = logits_from(kd)              # 缓存反解（含 fp8 重量化）
ker = rocm_fp8_paged_mqa_logits(q, kv4, w, ctx, bt, None, MML)
fal = fp8_paged_mqa_logits_torch(q, kv4, w, ctx, bt, MML)
def cmp(a, b, tag):
    fin = torch.isfinite(a) & torch.isfinite(b)
    x, y = a[fin].float(), b[fin].float()
    cos = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
    print("  %-28s cos=%.6f  最大差=%.4e" % (tag, cos, (x - y).abs().max().item()))
print("=== 以原始 k 为真值 ===")
cmp(ref_deq, ref_orig, "缓存反解(=真值)")
cmp(ker, ref_orig, "自研内核 vs 真值")
cmp(fal, ref_orig, "上游回退 vs 真值")
cmp(ker, ref_deq, "自研内核 vs 缓存反解")
print("=== 行 0 前 6 个 logits ===")
for nm, t in (("真值", ref_orig), ("缓存反解", ref_deq), ("内核", ker), ("回退", fal)):
    print("  %-8s %s" % (nm, [round(float(x), 3) for x in t[0, :6]]))
