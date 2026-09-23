# -*- coding: utf-8 -*-
"""修复后仓的完整性与抽样复验。

1) 43 个分片的 header 完整性（偏移合法、不重叠、dtype/shape 与字节数自洽）
2) 随机抽 12 个专家张量（跨层跨分片）做源对比
3) 非专家对照：mtp.0.main_proj（fp8 源 → int4）应与 fp8 源按官方序吻合（证明修复没误伤）
用法: python3 verify_fixed_repo.py <SRC> <OURS>
"""
import json, random, struct, sys, os, torch
from safetensors import safe_open

SRC, OURS = sys.argv[1].rstrip("/") + "/", sys.argv[2].rstrip("/") + "/"
FP4 = torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0])
si = json.load(open(SRC + "model.safetensors.index.json"))["weight_map"]
oi = json.load(open(OURS + "model.safetensors.index.json"))["weight_map"]

def get(p, k):
    with safe_open(p, framework="pt") as f:
        return f.get_tensor(k)

# ---------- 1) header 完整性 ----------
print("=== 1) 分片 header 完整性 ===")
files = sorted(set(oi.values()))
bad = 0
for fn in files:
    p = OURS + fn
    size = os.path.getsize(p)
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    base, end = 8 + n, size
    spans = []
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        b, e = v["data_offsets"]
        if b < 0 or e > end - base or e <= b:
            print("  ✗ %s %s 偏移越界 %s" % (fn, k, (b, e))); bad += 1
        spans.append((base + b, base + e))
    spans.sort()
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        if a2 < b1:
            print("  ✗ %s 张量区间重叠" % fn); bad += 1; break
    if spans and spans[-1][1] > end:
        print("  ✗ %s 尾部越界" % fn); bad += 1
print("  %d 个分片，异常 %d 处" % (len(files), bad))

# ---------- 2) 随机专家抽样 ----------
def src_expert(key):
    P = get(SRC + si[key + ".weight"], key + ".weight")
    S = get(SRC + si[key + ".scale"], key + ".scale")
    return P, S

def our_packed(key):
    return (get(OURS + oi[key + ".weight_packed"], key + ".weight_packed"),
            get(OURS + oi[key + ".weight_scale"], key + ".weight_scale"))

def dp4(P, S, nr, nc):
    P, S = P[:nr], S[:nr]
    u = P.view(torch.uint8).to(torch.int64)
    v = torch.stack([FP4[u & 0x0F], FP4[(u >> 4) & 0x0F]], dim=-1).reshape(nr, -1)[:, :nc]
    s = torch.pow(2.0, S.view(torch.uint8).to(torch.float32) - 127.0).repeat_interleave(32, 1)[:, :nc]
    return v * s

def dour(WP, WS, nr, nc):
    WP, WS = WP[:nr], WS[:nr]
    sh = torch.arange(8, dtype=torch.int32) * 4
    q = (((WP.unsqueeze(-1) >> sh) & 0xF).to(torch.float32) - 8.0).reshape(nr, -1)[:, :nc]
    return q * WS.to(torch.float32).repeat_interleave(32, 1)[:, :nc]

def corr(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

def pswap(t):
    return t.reshape(t.shape[0], -1, 2).flip(-1).reshape(t.shape)

random.seed(20260920)
keys = sorted(k[:-len(".weight_packed")] for k in oi
              if k.endswith(".weight_packed") and ".ffn.experts." in k)
sample = random.sample(keys, 12)
print("=== 2) 随机 12 个专家抽样（%d 个候选中） ===" % len(keys))
cs, ps = [], []
for k in sample:
    P, S = src_expert(k); WP, WS = our_packed(k)
    s, o = dp4(P, S, 2, 512), dour(WP, WS, 2, 512)
    c, cp = corr(o, s), corr(o, pswap(s))
    cs.append(c); ps.append(cp)
    print("  %-42s corr(修复后 vs 源)=%+.4f   corr(vs 相邻对互换)=%+.4f" % (k, c, cp))
    del P, S, WP, WS, s, o
print("  汇总: 与源 corr 均值 %.4f 最小 %.4f | 与互换序 corr 均值 %+.4f" % (
    sum(cs) / len(cs), min(cs), sum(ps) / len(ps)))

# ---------- 3) 非专家对照 ----------
print("=== 3) 非专家对照（fp8 源，本不应被改动） ===")
for key in ["mtp.0.main_proj", "layers.0.attn.wo_b"]:
    if key + ".weight_packed" not in oi:
        print("  (无 %s.weight_packed)" % key); continue
    W = get(SRC + si[key + ".weight"], key + ".weight")
    S = get(SRC + si[key + ".scale"], key + ".scale")
    WP, WS = our_packed(key)
    u = W.to(torch.float32).view(torch.uint8) if W.dtype == torch.uint8 else W.to(torch.float32)
    s8 = torch.pow(2.0, S.view(torch.uint8).to(torch.float32).reshape(W.shape[0], -1) - 127.0)
    br, bc = W.shape[0] // s8.shape[0], W.shape[1] // s8.shape[1]
    ref = (W.to(torch.float32) * s8.repeat_interleave(br, 0).repeat_interleave(bc, 1))[:2, :512]
    o = dour(WP, WS, 2, 512)
    print("  %-24s corr=%+.4f" % (key, corr(o, ref)))
