# -*- coding: utf-8 -*-
import json, struct, collections
S = "/mnt/kioxia-cm6-3t8/ai/models/GLM/GLM-5.3-753B/FP8"
O = "/mnt/kioxia-cm6-3t8/ai/models/GLM/GLM-5.3-753B/CT-Int4-W4A16"
si = json.load(open(S + "/model.safetensors.index.json"))["weight_map"]
oi = json.load(open(O + "/model.safetensors.index.json"))["weight_map"]

def dtypes(d, wm, mod):
    """返回 {后缀: dtype}"""
    out = {}
    fn = None
    for suf in ("weight", "weight_scale_inv", "weight_packed", "weight_scale", "weight_shape"):
        k = mod + "." + suf
        if k in wm:
            fn = wm[k] if fn is None else fn
    if fn is None:
        return {}
    with open(d + "/" + fn, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    for suf in ("weight", "weight_scale_inv", "weight_packed", "weight_scale", "weight_shape"):
        k = mod + "." + suf
        if k in h:
            out[suf] = h[k]["dtype"]
    return out

MODS = [
    "model.layers.10.mlp.shared_experts.down_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.2.mlp.up_proj",
    "model.layers.78.eh_proj",
    "model.layers.10.mlp.gate",
    "model.layers.0.self_attn.indexer.weights_proj",
    "model.layers.0.self_attn.q_b_proj",
    "model.layers.5.mlp.experts.3.gate_proj",
]
print("%-52s %-34s %s" % ("模块", "源", "产物"))
for m in MODS:
    print("%-52s %-34s %s" % (m.replace("model.layers.", ""),
                              dtypes(S, si, m), dtypes(O, oi, m)))
