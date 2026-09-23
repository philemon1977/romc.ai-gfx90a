#!/usr/bin/env python3
"""压缩池化核（Triton `fused_save_compress_norm`）的独立 torch 参考。

规格 = checkpoint 自带 inference/model.py 的 Compressor.forward（ratio>1 分支）：
    kv, score = wkv(x), wgate(x)                 # 都是 fp32
    pooled_g = sum_{i in group} kv_i * softmax(score_i over the group)
    latent_g = RMSNorm(pooled_g)                 # head_dim 维，eps=rms_norm_eps
**只写组边界行**；尾部不满一组的留在 state 里（不写）。
本脚本用运行时 dump 的 kv_score/positions/slot_mapping + 输出 latent 直接对拍，
把"池化 + 归一化"这一段从整条链里单独拎出来验。
"""
import json, sys
import torch

CKPT = "/models"
d = torch.load("/tmp/dsv41_cmp_in.pt", map_location="cpu", weights_only=False)
o = torch.load("/tmp/dsv41_cmp_out.pt", map_location="cpu", weights_only=False)
kvs = d["kv_score"].float()
pos = d["positions"].reshape(-1).to(torch.int64)
lat = o["latent"].float()
ratio = int(d["compress_ratio"]); hd = int(d["head_dim"])
eps = float(d["eps"]); nw = d["norm_weight"].float()
T = kvs.shape[0]
print(f"kv_score{tuple(kvs.shape)} positions={pos.tolist()[:24]} ratio={ratio} head_dim={hd} eps={eps:.1e}")
print(f"latent{tuple(lat.shape)}（只有组边界行被写；非边界行是空 buffer）")

assert kvs.shape[1] == 2 * hd, f"kv_score 第二维 {kvs.shape[1]} != 2*head_dim {2*hd}"
kv, score = kvs[:, :hd], kvs[:, hd:]        # [T,hd] | [T,hd]

# 参考池化：只在"整组都在本 chunk 内"时产出（与参考的 cutoff/remainder 语义一致）
start = int(pos[0].item())
seqlen = T
remainder = seqlen % ratio
cutoff = seqlen - remainder
rows, mine = [], []
for g0 in range(0, cutoff, ratio):
    g = slice(g0, g0 + ratio)
    w = torch.softmax(score[g], dim=0)
    pooled = (kv[g] * w).sum(dim=0, keepdim=True)
    mine.append(pooled)
    rows.append(g0 + ratio - 1)
mine = torch.cat(mine, dim=0) if mine else torch.zeros(0, hd)
rms = mine.pow(2).mean(-1, keepdim=True)
normed = mine * torch.rsqrt(rms + eps) * nw
print(f"参考算出的组数={len(rows)}  边界行位置={rows}（尾部 {remainder} 个 token 不满一组，按设计不写）")

fails = 0
print(f"\n{'pos':>4} {'|参考|max':>10} {'|latent|max':>11} {'maxabs差':>10} {'相对':>9}  判定")
for i, p in enumerate(rows):
    a = normed[i]; b = lat[p]
    dif = (a - b).abs().max().item()
    rel = dif / (b.abs().max().item() + 1e-9)
    ok = rel < 5e-3
    if not ok: fails += 1
    print(f"{p:>4} {a.abs().max():>10.4f} {b.abs().max():>11.4f} {dif:>10.3e} {rel:>9.2e}  {'PASS' if ok else '◆FAIL◆'}")
# 反向门：非边界行应当基本是 0（如果它也非零，说明写错了行）
nz = [(int(p), float(lat[int(p)].abs().max())) for p in range(T) if int(p) not in rows]
print(f"\n非边界行的 |latent| max（应≈0 或未写）: {nz[:8]}")
print(f"\n结论：{'池化+归一化与参考一致 ✅' if fails == 0 else f'{fails} 个边界行不一致 ❌ —— 压缩池化有问题'}")
