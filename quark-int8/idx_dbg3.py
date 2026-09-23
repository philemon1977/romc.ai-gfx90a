# -*- coding: utf-8 -*-
"""三方对拍定位差异：上游参考 vs 我的语义(torch) vs 我们的内核。"""
import os, torch
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    rocm_fp8_paged_mqa_logits, fp8_paged_mqa_logits_torch)
from vllm.v1.worker.workspace import init_workspace_manager
init_workspace_manager(torch.device("cuda"))
torch.manual_seed(0)

D, H, BS = 128, 32, 64
NB, B, CTXS, MML = 16, 2, [128, 200], 256
dev = "cuda"
k = torch.randn(NB*BS, D, device=dev) * 0.15
sc = torch.rand(NB*BS, device=dev) * 0.5 + 0.75
kv = torch.zeros(NB, BS * D + BS * 4, dtype=torch.uint8, device=dev)
kv[:, : BS * D] = k.to(torch.float8_e4m3fn).view(torch.uint8).view(NB, BS * D)
kv[:, BS * D :] = sc.view(torch.float32).view(torch.uint8).view(NB, BS * 4)
kv = kv.view(NB, BS, 1, D + 4)   # 生产同形 4-D 视图（页内布局 = 值区 + scale 区）
q = (torch.randn(B, 1, H, D, device=dev)*0.15).to(torch.float8_e4m3fn)
w = torch.rand(B, H, device=dev)
ctx = torch.tensor(CTXS, dtype=torch.int32, device=dev)
bt = torch.arange(NB, dtype=torch.int32, device=dev).view(1, -1).repeat(B, 1)

ref = fp8_paged_mqa_logits_torch(q, kv, w, ctx, bt, MML)
ker = rocm_fp8_paged_mqa_logits(q, kv, w, ctx, bt, None, MML)

# 我的语义（torch 复算，不用 kernel）
qf = q.float()                                    # [B,1,H,D]
kvflat = kv.view(NB, -1)
kf = kvflat[:, : BS * D].view(torch.float8_e4m3fn).float().view(NB, BS, D)
scf = kvflat[:, BS * D :].contiguous().view(torch.float32).view(NB, BS)   # 每位置 scale
mine = torch.full((B, MML), float("-inf"), device=dev)
for i in range(B):
    c = int(ctx[i])
    qq = qf[i, 0]                                  # [H,D]
    allk = kf.reshape(NB*BS, D)[:c]                # 前 c 个位置（bt 是恒等映射）
    s = torch.relu(qq @ allk.T) * w[i][:, None]    # [H, c]
    mine[i, :c] = s.sum(0) * scf.reshape(-1)[:c]

def cmp(a, b, tag):
    fin = torch.isfinite(a) & torch.isfinite(b)
    same_mask = torch.equal(torch.isfinite(a), torch.isfinite(b))
    if fin.any():
        x, y = a[fin].float(), b[fin].float()
        cos = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
        print("  %-18s 掩码一致=%-5s 最大差=%.4e cos=%.6f" % (tag, same_mask, (x-y).abs().max().item(), cos))
    else:
        print("  %-18s 无公共有限元" % tag)
print("=== 三方对拍 ===")
cmp(ker, ref, "kernel vs 参考")
cmp(mine, ref, "mine(torch) vs 参考")
cmp(ker, mine, "kernel vs mine")
print("=== 行0 前 8 个 logits ===")
for name, t in (("ref", ref), ("kernel", ker), ("mine", mine)):
    print("  %-7s %s" % (name, [round(float(x), 3) for x in t[0, :8]]))
print("=== 有效长度（有限元个数/行） ===")
print("  ref   ", [int(torch.isfinite(ref[i]).sum()) for i in range(B)])
print("  kernel", [int(torch.isfinite(ker[i]).sum()) for i in range(B)])
print("  mine  ", [int(torch.isfinite(mine[i]).sum()) for i in range(B)])