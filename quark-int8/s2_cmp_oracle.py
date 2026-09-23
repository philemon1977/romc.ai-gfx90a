#!/usr/bin/env python3
# S2-CMP oracle：按官方 Compressor 规格离线复算池化，对拍 fused_save_compress_norm 的 latent。
# 官方（ratio>1, prefill, start_pos=0）：kv, score = split(kv_score)；
#   每组 ratio 个 token：latent_g = sum(kv * softmax(score, dim=g))；再 RMSNorm(eps) 到 bf16。
# 变体 = (前半=kv|score) × (latent 行映射：token 位=组末 / 组序顺排)。
import sys, torch

inp = torch.load(sys.argv[1] if len(sys.argv) > 1 else "cmp_dump/dsv41_cmp_in.pt", map_location="cpu")
outp = torch.load(sys.argv[2] if len(sys.argv) > 2 else "cmp_dump/dsv41_cmp_out.pt", map_location="cpu")
kvs = inp["kv_score"].float()          # [T, 2d] 或 [T, d]（ratio==1 只有 kv）
pos = inp["positions"].long()          # [T] token 全局位置
r = int(inp["compress_ratio"]); d = int(inp["head_dim"])
w = inp["norm_weight"].float(); eps = float(inp["eps"])
lat = outp["latent"].float()           # 运行时输出缓冲
print("shapes: kv_score=%s pos=%s ratio=%d head_dim=%d latent=%s layer=%s" % (
    tuple(kvs.shape), tuple(pos.shape), r, d, tuple(lat.shape), inp.get("prefix")))

def rmsnorm(x):
    v = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + eps) * w

T = kvs.shape[0]
if kvs.shape[-1] == 2 * d:
    variants = [("first=kv", kvs[:, :d], kvs[:, d:]), ("first=score", kvs[:, d:], kvs[:, :d])]
else:
    variants = [("ratio1-nopool", kvs, None)]

def groups_of(mask_end):
    # 完整组：组内 ratio 个 token 都在本次调用里（按 pos//r 分组）
    g = {}
    for i in range(T):
        g.setdefault(int(pos[i]) // r, []).append(i)
    return {k_: v_ for k_, v_ in g.items() if len(v_) == r}

best = None
for name, kv, sc in variants:
    if sc is None:
        exp = rmsnorm(kv)                        # 每 token 一个 latent
        mapA = exp
        mapB = None
    else:
        gs = groups_of(None)
        pooled = []
        for g_ in sorted(gs):
            idx = torch.tensor(gs[g_])
            sft = torch.softmax(sc[idx].float(), dim=0)
            pooled.append((kv[idx] * sft).sum(0))
        expg = rmsnorm(torch.stack(pooled)) if pooled else None
        if expg is None:
            continue
        mapB = expg                              # 组序顺排
        mapA = torch.zeros(T, d)                 # token 位=组末
        for gi, g_ in enumerate(sorted(gs)):
            mapA[gs[g_][-1]] = mapB[gi]
    for tag, cand in (("rowmap=token_end", mapA), ("rowmap=seq", mapB)):
        if cand is None:
            continue
        m = min(cand.shape[0], lat.shape[0])
        nz = lat[:m].abs().sum(-1) > 0
        if int(nz.sum()) < 1:
            continue
        a, b = cand[:m][nz], lat[:m][nz]
        corr = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
        rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-9)
        rec = (corr, rel, name + " " + tag)
        print("%-14s %-16s rows=%d corr=%.6f rel=%.5f" % (name, tag, int(nz.sum()), corr, rel))
        if best is None or rec[0] > best[0]:
            best = rec
print()  
print("BEST:", best)
print("判定：%s" % ("CORR≈1 ⇒ fused 核与官方规格一致（该层语义无罪）" if best and best[0] > 0.9999 else "全变体都对不上 ⇒ 定罪 fused_save_compress_norm(ROCm 分支)")),