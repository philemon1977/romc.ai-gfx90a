# -*- coding: utf-8 -*-
"""GLM-5.3 分类参数量与量化后体积估算（决定能否装进 512GB HBM）。"""
import json, os, re, struct, collections

M = "/mnt/kioxia-cm6-3t8/ai/models/GLM/GLM-5.3-753B/FP8"
idx = json.load(open(os.path.join(M, "model.safetensors.index.json")))["weight_map"]
files = sorted(set(idx.values()))

DT = {"F8_E4M3": 1, "BF16": 2, "F32": 4, "I8": 1, "U8": 1, "I32": 4, "I64": 8}

def klass(k):
    if re.search(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$", k):
        return "专家(路由)"
    if ".mlp.shared_experts." in k:
        return "共享专家"
    if ".self_attn.indexer." in k:
        return "DSA indexer"
    if ".self_attn." in k:
        return "注意力"
    if ".mlp.gate" in k:
        return "路由器"
    if "embed_tokens" in k or "lm_head" in k:
        return "词嵌入/输出头"
    if "nextn" in k or "mtp" in k:
        return "MTP"
    return "其它(norm等)"

elems = collections.Counter()   # 类别 -> 元素数（仅 .weight）
scale_elems = collections.Counter()
for fn in files:
    with open(os.path.join(M, fn), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    for k, v in hdr.items():
        if k == "__metadata__" or "scale" in k:
            continue
        nb = v["data_offsets"][1] - v["data_offsets"][0]
        el = DT.get(v["dtype"], 1)
        elems[klass(k)] += nb // el

tot = sum(elems.values())
print("=== 元素数（按类别） ===")
for c, e in elems.most_common():
    print("   %-14s %14d  (%5.1f%%)" % (c, e, 100.0 * e / tot))
print("   %-14s %14d" % ("合计", tot))
print()
G = 2**30
q = lambda e: e * (0.5 + 2 / 32) / G      # int4 + fp16 g32 scale
b = lambda e: e * 2 / G                   # bf16
print("=== 方案体积估算（GiB） ===")
exp = elems["专家(路由)"]
plan = {
    "A 全 int4（专家+注意力+共享专家+indexer 都 int4）": q(exp + elems["注意力"] + elems["共享专家"] + elems["DSA indexer"]) + b(elems["路由器"] + elems["词嵌入/输出头"] + elems["其它(norm等)"]),
    "B 注意力 bf16（专家 int4，其它 bf16）": q(exp) + b(elems["注意力"] + elems["共享专家"] + elems["DSA indexer"] + elems["路由器"] + elems["词嵌入/输出头"] + elems["其它(norm等)"]),
    "C 专家 int4 + 注意力 int4（共享专家/路由器/norm bf16）": q(exp + elems["注意力"] + elems["DSA indexer"]) + b(elems["共享专家"] + elems["路由器"] + elems["词嵌入/输出头"] + elems["其它(norm等)"]),
    "（对照）全 int8：": (tot) / G,
}
for name, gib in plan.items():
    left = 496 - gib
    print("   %-52s %7.1f GiB   余量 %6.1f GiB %s" % (name, gib, left, "✓ 装得下" if left > 20 else "✗ 装不下"))
print()
print("说明：可用 HBM = 8 × 62 GiB ≈ 496 GiB（util 0.97 减 HIP 上下文）")
