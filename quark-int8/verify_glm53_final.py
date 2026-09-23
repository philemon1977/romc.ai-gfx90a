# -*- coding: utf-8 -*-
"""最终产物保真度复验（独立尺子，按类别抽样）：
  1) weight_shape 是否 == 源 [out, in]（抓转置/形状错）
  2) int4 反量化 vs 源 fp8 反量化：cos / rel / max|err|/amax
"""
import json, random, sys, torch
from safetensors import safe_open

S = sys.argv[1].rstrip("/")
O = sys.argv[2].rstrip("/")
si = json.load(open(S + "/model.safetensors.index.json"))["weight_map"]
oi = json.load(open(O + "/model.safetensors.index.json"))["weight_map"]

def get(d, k):
    with safe_open(d + "/" + (si if d == S else oi)[k], framework="pt") as f:
        return f.get_tensor(k)

def src_dense(w, s):
    s = s.to(torch.float32)
    n, k = w.shape
    blk = 128
    s = s.repeat_interleave(blk, 0)[:n].repeat_interleave(blk, 1)[:, :k]
    return w.to(torch.float32) * s

def our_dense(wp, ws, group=32):
    sh = torch.arange(8, dtype=torch.int32) * 4
    q = ((wp.unsqueeze(-1) >> sh) & 0xF).to(torch.float32) - 8.0
    q = q.reshape(wp.shape[0], -1)
    return q * ws.to(torch.float32).repeat_interleave(group, dim=1)

random.seed(20260920)
packed = [k for k in oi if k.endswith(".weight_packed")]
cls = {
    "专家": [k for k in packed if ".mlp.experts." in k],
    "注意力": [k for k in packed if ".self_attn." in k and "indexer" not in k],
    "indexer.wq_b": [k for k in packed if "indexer.wq_b" in k],
    "dense MLP": [k for k in packed if k.split(".mlp.")[-1].startswith(("gate_proj", "up_proj", "down_proj")) and "experts" not in k],
}
print("%-14s %-46s %-24s %s" % ("类别", "模块", "weight_shape vs 源", "cos / rel / max|err|/amax"))
for name, keys in cls.items():
    if not keys:
        print("  %-12s (无)" % name); continue
    for k in random.sample(keys, min(2, len(keys))):
        mod = k[: -len(".weight_packed")]
        sk = mod + ".weight"
        ss = mod + ".weight_scale_inv"
        if sk not in si:
            print("  %-12s %-46s 源缺 weight" % (name, mod[-46:])); continue
        W = get(S, sk); Ss = get(S, ss)
        ws = get(O, mod + ".weight_shape")
        shape_ok = (tuple(ws.tolist()) == tuple(W.shape))
        nr = 4
        a = src_dense(W[:nr], Ss[:nr]).float()
        b = our_dense(get(O, k)[:nr], get(O, mod + ".weight_scale")[:nr]).float()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        rel = ((a - b).norm() / a.norm()).item()
        amax = a.reshape(a.shape[0], -1, 32).abs().amax(-1, keepdim=True)
        err = (a - b).reshape(a.shape[0], -1, 32)
        bound = (err.abs() / amax.clamp_min(1e-9)).max().item()
        print("  %-12s %-46s %-24s cos=%.6f rel=%.4f bound=%.4f"
              % (name, mod[-46:], "%s %s" % (tuple(ws.tolist()), "OK" if shape_ok else "MISMATCH vs %s" % (tuple(W.shape),)), cos, rel, bound))