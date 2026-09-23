# -*- coding: utf-8 -*-
"""找出 weight / weight_scale_inv 形状不一致的模块（试点 ZeroDivisionError 的来源）。"""
import json, struct, sys, collections
M = "/mnt/kioxia-cm6-3t8/ai/models/GLM/GLM-5.3-753B/FP8"
idx = json.load(open(M + "/model.safetensors.index.json"))["weight_map"]
bad = []
for fn in ["model-00001-of-00141.safetensors", "model-00002-of-00141.safetensors"]:
    with open(f"{M}/{fn}", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    for k, v in h.items():
        if not k.endswith(".weight"):
            continue
        sk = k[: -len(".weight")] + ".weight_scale_inv"
        if sk not in h:
            continue
        ws, ss = v["shape"], h[sk]["shape"]
        if len(ss) != 2 or ss[0] == 0 or ws[0] % ss[0] or ws[1] % ss[1]:
            bad.append((k, tuple(ws), tuple(ss), h[sk]["dtype"]))
print("形状不自洽的模块数:", len(bad))
for b in bad[:10]:
    print("   ", b)
# 也统计一下 scale 的行块大小分布
sizes = collections.Counter()
for fn in ["model-00001-of-00141.safetensors", "model-00002-of-00141.safetensors"]:
    with open(f"{M}/{fn}", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    for k, v in h.items():
        if k.endswith(".weight"):
            sk = k[: -len(".weight")] + ".weight_scale_inv"
            if sk in h and len(h[sk]["shape"]) == 2 and h[sk]["shape"][0]:
                sizes[(v["shape"][0] // h[sk]["shape"][0], v["shape"][1] // h[sk]["shape"][1])] += 1
print("块大小分布 (行块, 列块):", sizes.most_common(5))
