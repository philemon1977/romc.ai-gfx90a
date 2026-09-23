# -*- coding: utf-8 -*-
"""FP4 nibble-order audit.

官方 convert.py（随模型目录发布）与运行时解包都规定:
    byte b:  low nibble = element 2b ,  high nibble = element 2b+1
而我们的 convert_dsv41_ct_int4.py::dequant_fp4_expert 用的是:
    hi = element 2j , lo = element 2j+1   == 相邻对互换
本脚本判定 CT 仓中的 int4 专家是否为源行的「相邻对互换」版本。
用法: python3 audit_fp4_nibble_order.py <SRC> <OURS>
"""
import sys, json, torch
from safetensors import safe_open

SRC, OURS = sys.argv[1].rstrip("/") + "/", sys.argv[2].rstrip("/") + "/"
si = json.load(open(SRC + "model.safetensors.index.json"))["weight_map"]
oi = json.load(open(OURS + "model.safetensors.index.json"))["weight_map"]
FP4 = torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0])

def get(p, k):
    with safe_open(p, framework="pt") as f:
        return f.get_tensor(k)

def src_dense(P, S, nr, nc):
    P = P[:nr]; S = S[:nr]
    u = P.view(torch.uint8).to(torch.int64)
    lo = FP4[u & 0x0F]; hi = FP4[(u >> 4) & 0x0F]
    v = torch.stack([lo, hi], dim=-1).reshape(nr, -1)[:, :nc]
    s = torch.pow(2.0, S.view(torch.uint8).to(torch.float32) - 127.0).repeat_interleave(32, dim=1)[:, :nc]
    return v * s

def our_dense(WP, WS, nr, nc):
    WP = WP[:nr]; WS = WS[:nr]
    sh = torch.arange(8, dtype=torch.int32) * 4
    q = ((WP.unsqueeze(-1) >> sh) & 0xF).to(torch.float32) - 8.0
    q = q.reshape(nr, -1)[:, :nc]
    s = WS.to(torch.float32).repeat_interleave(32, dim=1)[:, :nc]
    return q * s

def pswap(t):
    return t.reshape(t.shape[0], -1, 2).flip(-1).reshape(t.shape)

def stats(a, b, tag):
    a = a.float().flatten(); b = b.float().flatten()
    c = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    rel = ((a - b).norm() / b.norm()).item()
    print("   %-26s corr=%+.4f  relL2=%.4f" % (tag, c, rel))

for (L, E, W) in [(0, 0, "w1"), (3, 5, "w1"), (0, 0, "w2"), (7, 13, "w3")]:
    sk = "layers.%d.ffn.experts.%d.%s" % (L, E, W)
    try:
        P = get(SRC + si[sk + ".weight"], sk + ".weight")
        S = get(SRC + si[sk + ".scale"], sk + ".scale")
        WP = get(OURS + oi[sk + ".weight_packed"], sk + ".weight_packed")
        WS = get(OURS + oi[sk + ".weight_scale"], sk + ".weight_scale")
    except KeyError as e:
        print("skip", sk, e); continue
    NR, NC = 4, 1024
    s = src_dense(P, S, NR, NC); o = our_dense(WP, WS, NR, NC)
    print("== %s src%s our%s (源 packed%s) ==" % (sk, tuple(s.shape), tuple(o.shape), tuple(P.shape)))
    stats(o, s, "our vs src(官方 low=even)")
    stats(o, pswap(s), "our vs src(相邻对互换)")
    del P, S, WP, WS, s, o
