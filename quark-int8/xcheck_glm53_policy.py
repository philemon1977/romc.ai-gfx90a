# -*- coding: utf-8 -*-
"""交叉验证：我们的 keep 集合 vs 上游 FP8 配置的 modules_to_not_convert。

上游作者用 modules_to_not_convert 声明"这些模块不做 fp8 量化"——
如果我们的"保持未量化"集合与它一致，说明精度策略与上游意图吻合；
差集里的 Linear 类模块就是"可能会在装载时炸"的候选。
"""
import json, re, sys

S = "/mnt/kioxia-cm6-3t8/ai/models/GLM/GLM-5.3-753B/FP8"
O = "/mnt/kioxia-cm6-3t8/ai/models/GLM/GLM-5.3-753B/CT-Int4-W4A16"

up = set(json.load(open(S + "/config.json"))["quantization_config"]["modules_to_not_convert"])
oi = json.load(open(O + "/model.safetensors.index.json"))["weight_map"]

# 我们的产物里"未量化"的模块 = 有普通 .weight 且没有 .weight_packed
packed = {k[: -len(".weight_packed")] for k in oi if k.endswith(".weight_packed")}
ours = {k[: -len(".weight")] for k in oi
        if k.endswith(".weight") and not k.endswith(".weight_scale_inv")
        and k[: -len(".weight")] not in packed}

print("上游 not-convert : %d" % len(up))
print("我们未量化       : %d" % len(ours))
norm = lambda s: {re.sub(r"\.\d+\.", ".N.", x) for x in s}
print()
print("=== 上游有、我们没有（应为我们额外量化了 ⇒ 需确认它们是 Linear）===")
d1 = sorted(norm(up) - norm(ours))
print("  模式数 %d" % len(d1))
for x in d1[:12]:
    print("   ", x)
print()
print("=== 我们有、上游没有（我们保守保留 ⇒ 数量应==", len(sorted(norm(ours) - norm(up))), "）===")
d2 = sorted(norm(ours) - norm(up))
for x in d2[:12]:
    print("   ", x)
