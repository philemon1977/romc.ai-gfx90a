#!/usr/bin/env python3
# S2 oracle v2：按官方 inference/model.py 规格复算 indexer logits，对拍 ④ dump。
# 变体 = (scale 施加位置: k侧/无) × (relu: 有/无)。
# 有任一变体全层 corr>0.9999 且 rel<1e-2 ⇒ 该层 logits 语义成立；全灭 ⇒ S2 定罪。
import glob, os, sys, torch

DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/idx_dump"

def dequant(x, sc, block):
    if sc is None:
        return x
    if sc.dim() == 1:
        sc = sc.view(1, -1).expand(x.shape[0], -1)
    s = torch.pow(2.0, sc.to(torch.float32) - 127.0)
    return x.view(x.shape[0], -1, block).mul(s.unsqueeze(-1)).view(x.shape[0], -1)

names = []
for path in sorted(glob.glob(os.path.join(DIR, "layer_*.pt"))):
    d = torch.load(path, map_location="cpu")
    q, k, w = d["q_fp8"], d["k_fp8"], d["weights"]
    rt = d["logits"].float()
    if rt.dim() != 2:
        rt = rt.reshape(rt.shape[0], -1)
    if q.dim() != 3:
        print(os.path.basename(path), "SKIP q.dim=", q.dim()); continue
    T = min(rt.shape[1], k.shape[0])
    rtv = rt[:, :T]
    ke = d["cu_ke"].long().clamp(max=T)
    ks = d["cu_ks"].long().clamp(max=T)
    block = max(int(d.get("quant_block_size", 128) or 128), 1)
    ksc = d.get("k_scale")
    m = torch.zeros_like(rtv, dtype=torch.bool)
    n_rows = min(rtv.shape[0], ke.numel())
    for i in range(n_rows):
        a, b = int(ks[i]), int(ke[i])
        if b > a:
            m[i, a:b] = True
    if int(m.sum()) == 0:
        print(os.path.basename(path), "SKIP empty mask"); continue
    kk = k[:T]
    res = []
    for kmode in ("none", "scale"):
        if kmode == "scale" and ksc is None:
            continue
        kv = dequant(kk, ksc, block) if kmode == "scale" else kk
        if kv.shape[0] != T:
            continue
        s = torch.einsum("shd,td->sht", q.float(), kv.float())
        for relumode in ("relu", "raw"):
            sx = s.relu() if relumode == "relu" else s
            o = (sx * w.float().unsqueeze(-1)).sum(dim=2)[:, :T]
            aa, bb = o[m], rtv[m]
            rel = (aa - bb).abs().max().item() / max(bb.abs().max().item(), 1e-9)
            corr = torch.corrcoef(torch.stack([aa, bb]))[0, 1].item()
            res.append(((kmode, relumode), rel, corr))
    res.sort(key=lambda v: -v[2])
    best = res[0]
    names.append(os.path.basename(path))
    print("%-28s shape=%-12s pts=%-6d BEST k=%-5s %s rel=%.5f corr=%.6f" % (
        os.path.basename(path), str(tuple(rtv.shape)), int(m.sum()),
        best[0][0], best[0][1], best[1], best[2]))
    if len(res) > 1:
        wv = res[-1]
        print("%-28s worst: k=%-5s %s rel=%.4f corr=%.6f" % ("", wv[0][0], wv[0][1], wv[1], wv[2]))
print("layers checked:", len(names))