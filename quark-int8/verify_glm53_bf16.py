# -*- coding: utf-8 -*-
"""复验 fp8 → bf16 的两类转换（共享专家 / indexer.wk）：应与源 fp8 反量化几乎逐元素相等。"""
import json, sys, torch
from safetensors import safe_open
S, O = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
si = json.load(open(S + "/model.safetensors.index.json"))["weight_map"]
oi = json.load(open(O + "/model.safetensors.index.json"))["weight_map"]

def get(d, wm, k):
    with safe_open(d + "/" + wm[k], framework="pt") as f:
        return f.get_tensor(k)

CASES = sorted([k[: -len(".weight")] for k in oi
                if "shared_experts" in k and k.endswith(".weight")])[:2] + \
        sorted([k[: -len(".weight")] for k in oi
                if "indexer.wk" in k and k.endswith(".weight")])[:2]
print("%-52s %-8s %s" % ("模块", "产物", "vs 源 fp8 反量化"))
for mod in CASES:
    w = get(O, oi, mod + ".weight")
    sw = get(S, si, mod + ".weight")
    ss = get(S, si, mod + ".weight_scale_inv")
    s = ss.to(torch.float32).repeat_interleave(128, 0).repeat_interleave(128, 1)[: sw.shape[0], : sw.shape[1]]
    ref = (sw.to(torch.float32) * s)
    a, b = w[:4].float(), ref[:4]
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rel = ((a - b).norm() / b.norm()).item()
    print("  %-50s %-8s cos=%.8f rel=%.6f  (bf16 舍入量级 ≈ 0.004)" % (mod[-50:], str(w.dtype).replace("torch.", ""), cos, rel))
