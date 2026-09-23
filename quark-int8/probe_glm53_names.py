# -*- coding: utf-8 -*-
"""GLM-5.3 张量命名/比例盘点：决定转换器改造范围与 int4 后体积估算。"""
import json, os, collections, struct, re

M = "/mnt/kioxia-cm6-3t8/ai/models/GLM/GLM-5.3-753B/FP8"
idx = json.load(open(os.path.join(M, "model.safetensors.index.json")))["weight_map"]
pat = collections.Counter()
for k in idx:
    kk = re.sub(r"\.\d+", ".N", k)
    pat[kk] += 1
print("=== 张量名模式（去掉层号/专家号）top 25 ===")
for k, c in pat.most_common(25):
    print("   %6d  %s" % (c, k))
print()
print("=== 含 experts 的键样例 ===")
ex = [k for k in idx if "experts" in k]
print("   experts 张量数:", len(ex))
for k in ex[:6]:
    print("   ", k)
print()
print("=== 含 attn 的键样例 ===")
at = [k for k in idx if "self_attn" in k]
print("   张量数:", len(at))
for k in at[:10]:
    print("   ", k)
