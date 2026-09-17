#!/usr/bin/env python3
"""离线二分：我的内核 vs 参考，逐对比较；并把 scale 去掉再比，判断是否为 scale 语义问题。"""
import torch, sys
sys.path.insert(0, "/home/qiba/ROCm.AI/quark-int8")
from gemv_moe import run_ours
from repro_offline import GROUP

d = torch.load("/tmp/mi250_moe_real.pt", map_location="cuda")
A, B, S = d["A"].cuda(), d["B"].cuda(), d["B_scale"].cuda()
ids = d["topk_ids"].reshape(-1).cuda()
N, K, top_k = d["N"], d["K"], d["top_k"]
Ntok = A.shape[0]; pairs = Ntok * top_k
used = torch.unique(ids[:pairs]).tolist(); remap = {e: i for i, e in enumerate(used)}

wb = B[used].to(torch.int64)
nib = torch.empty(len(used), N, K, dtype=torch.float32, device="cuda")
for j in range(2):
    nib[:, :, j::2] = ((wb >> (4 * j)) & 0xF).to(torch.float32) - 8.0
sc_full = S[used].to(torch.float32).repeat_interleave(GROUP, dim=2)
wdq = nib * sc_full
x = A.to(torch.float32)

ref = torch.stack([wdq[remap[int(ids[p])]] @ x[p // top_k] for p in range(pairs)])
ref_ns = torch.stack([nib[remap[int(ids[p])]] @ x[p // top_k] for p in range(pairs)])

mine = run_ours(A, B, S, ids[:pairs], torch.ones(pairs, device="cuda"), N, top_k,
                BLOCK_N=128, BLOCK_K=128, warps=4).float()

print(f"pairs={pairs}  N={N} K={K}  used_experts={len(used)}")
print(f"|ref|={ref.abs().mean():.5f}   |ref_noscale|={ref_ns.abs().mean():.5f}   |mine|={mine.abs().mean():.5f}")
print(f"mine vs ref       : {(mine-ref).abs().mean()/ref.abs().mean()*100:.2f}%")
print(f"mine vs ref_noscale: {(mine-ref_ns).abs().mean()/ref_ns.abs().mean()*100:.2f}%")
r = (mine.abs().mean() / ref.abs().mean()).item()
print(f"|mine|/|ref| = {r:.3f}    |ref|/|ref_nos| = {(ref.abs().mean()/ref_ns.abs().mean()).item():.3e}")
print("\n每个专家的 scale 量级（前 5 个 used）:")
for e in used[:5]:
    print(f"  expert {e}: S 均值={S[e].float().mean():.3e} 最大={S[e].float().max():.3e}")
print("\n逐对前 8 个比值 mine/ref:")
for p in range(8):
    a, b = mine[p].abs().mean().item(), ref[p].abs().mean().item()
    print(f"  pair{p}: mine={a:.5f} ref={b:.5f} 比值={a/max(b,1e-12):.3f}")
