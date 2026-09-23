# -*- coding: utf-8 -*-
"""审查：复审报出的三类问题，逐个查源/产物的 dtype 与键，判断哪些是真问题。"""
import json, struct, sys, collections

S = "/mnt/kioxia-cm6-3t8/ai/models/GLM/GLM-5.3-753B/FP8"
O = "/mnt/kioxia-cm6-3t8/ai/models/GLM/GLM-5.3-753B/CT-Int4-W4A16"

def idx(d):
    return json.load(open(d + "/model.safetensors.index.json"))["weight_map"]

si, oi = idx(S), idx(O)

def dtypes_of(d, wm, mods):
    """按需读取分片头，返回 {tensor: dtype}"""
    need = collections.defaultdict(list)
    for m in mods:
        for suf in ("weight", "weight_scale_inv", "weight_packed", "weight_scale", "weight_shape"):
            k = m + "." + suf
            if k in wm:
                need[wm[k]].append(k)
    out = {}
    for fn, ks in need.items():
        with open(d + "/" + fn, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        for k in ks:
            out[k] = (h[k]["dtype"], tuple(h[k]["shape"]))
    return out

print("=== 1) 层 78（MTP）全部模块 ===")
l78 = sorted({k.rsplit(".", 1)[0] for k in si if k.startswith("model.layers.78.")})
for m in l78:
    sk = si.get(m + ".weight")
    print("   %-58s src_weight=%-9s scale=%s" % (m.replace("model.layers.78.", "78."),
          (dtypes_of(S, si, [m]).get(m + ".weight") or ("-",))[0],
          "有" if m + ".weight_scale_inv" in si else "无"))

print()
print("=== 2) 共享专家：源 fp8，产物是否已 dequant 成 bf16 ===")
m = "model.layers.10.mlp.shared_experts.down_proj"
print("   源 :", dtypes_of(S, si, [m]))
print("   产物:", dtypes_of(O, oi, [m]))

print()
print("=== 3) dense MLP（重转后）===")
for m in ("model.layers.0.mlp.down_proj", "model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"):
    print("   %-40s 产物=%s" % (m, {k.split(".")[-1]: v[0] for k, v in dtypes_of(O, oi, [m]).items()}))

print()
print("=== 4) 路由器（应保持未量化）===")
for m in ("model.layers.10.mlp.gate", "model.layers.0.mlp.gate"):
    print("   %-40s 源=%s 产物=%s" % (m, {k.split(".")[-1]: v[0] for k, v in dtypes_of(S, si, [m]).items()},
                                      {k.split(".")[-1]: v[0] for k, v in dtypes_of(O, oi, [m]).items()}))
