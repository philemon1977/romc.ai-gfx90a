# -*- coding: utf-8 -*-
"""转换保真度复验：我们的 int4（weight_packed + weight_scale）反量化 vs 源 fp8 块量化反量化。

独立第三方尺子：源侧用官方 fp8 块语义（w * 2^(e8m0-127)，128x128 块），
我们侧用 CT 约定（int32 低 nibble 先 = K 的第 0 个元素，value = nibble-8，组内乘 scale）。
两次都独立于转换器的内部函数，避免"闭环自证"。
"""
import json, sys, torch
from safetensors import safe_open

M, O = sys.argv[1].rstrip("/") + "/", sys.argv[2].rstrip("/") + "/"
si = json.load(open(M + "model.safetensors.index.json"))["weight_map"]
oi = json.load(open(O + "model.safetensors.index.json"))["weight_map"]

def get(p, k):
    with safe_open(p, framework="pt") as f:
        return f.get_tensor(k)

def src_dense(w, s):
    # GLM-5.3 的 weight_scale_inv 是 F32 乘数；DSV4.1 是 F8_E8M0 指数字节
    if s.dtype in (torch.uint8, torch.float8_e8m0fnu):
        s = torch.pow(2.0, s.view(torch.uint8).to(torch.float32) - 127.0)
    else:
        s = s.to(torch.float32)
    br, bc = w.shape[0] // s.shape[0], w.shape[1] // s.shape[1]
    return (w.to(torch.float32) * s.repeat_interleave(br, 0).repeat_interleave(bc, 1))

def our_dense(wp, ws, group=32):
    sh = torch.arange(8, dtype=torch.int32) * 4
    q = ((wp.unsqueeze(-1) >> sh) & 0xF).to(torch.float32) - 8.0
    q = q.reshape(wp.shape[0], -1)
    return q * ws.to(torch.float32).repeat_interleave(group, dim=1)

keys = [k for k in oi if k.endswith(".weight_packed")]
print("产物 int4 张量数:", len(keys))
import random
random.seed(7)
sample = random.sample(keys, min(10, len(keys)))
cs, rels, bounds = [], [], []
for k in sample:
    mod = k[: -len(".weight_packed")]
    sw, ss = mod + ".weight", mod + ".weight_scale_inv"
    if sw not in si:
        print("  skip(源无此张量):", mod); continue
    a = src_dense(get(M + si[sw], sw), get(M + si[ss], ss))
    b = our_dense(get(O + oi[k], k), get(O + oi[mod + ".weight_scale"], mod + ".weight_scale"))
    n = min(a.shape[0], 8)
    a, b = a[:n].float(), b[:n].float()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rel = ((a - b).norm() / a.norm()).item()
    # 理论界：对称 int4 组内 |err| <= amax/14
    g = a.reshape(a.shape[0], -1, 32)
    amax = g.abs().amax(-1, keepdim=True)
    err = (a - b).reshape(a.shape[0], -1, 32)
    bound = (err.abs() / amax.clamp_min(1e-9)).max().item()
    cs.append(cos); rels.append(rel); bounds.append(bound)
    print("  %-58s cos=%.6f rel=%.4f max|err|/amax=%.4f" % (mod[-58:], cos, rel, bound))
print("汇总: cos 最小 %.6f | rel 中位 %.4f | max|err|/amax 最大 %.4f（理论界 1/14=0.0714）"
      % (min(cs), sorted(rels)[len(rels)//2], max(bounds)))
