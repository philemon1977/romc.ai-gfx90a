#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按类别算体积账：把 fp8 源那部分（注意力/共享专家/索引器/压缩器）留 bf16，代价到底多大？

## 动机（来自 v67 的功能损伤分解）

layer-0 MoE 输出相对理想（源权重）的偏差 9.30%，而该层输出量级是
    routed rms ≈ 0.0184   shared rms ≈ 0.0930    ← 共享专家占 5 倍
共享专家是 **fp8 源**、被我们转成 int4(g32)，而 fp8 源的最优 scale 只有 1.13× 收益
⇒ **功能损伤几乎全部来自 fp8 源那部分的 int4 化**，而不是路由专家。

于是关键问题变成"代价"：注意力/共享专家等 fp8 张量在**整仓数值量**里占多少？
若只有百分之一二，就完全可以把它们留 bf16（甚至 fp8），只把路由专家 int4 化，
用 1~2 GiB/rank 换掉主要的损伤来源。

## 口径

只读 48 个分片的 header（不读数据），按名字分类求和元素数：
  fp4_expert : layers.*.ffn.experts.*.{w1,w2,w3}.weight   （源为 fp4，1 值占 4bit+scale）
  fp8_block  : attn.*、indexer.*、shared_experts.*、compressor.* 等（源为 fp8 1 字节）
  其它       : norm / gate / embed / head / engram …
体积口径：
  源 fp4  = 0.5 B/值 + e8m0 scale(1B/32值) = 0.53125 B/值
  我们 int4= 0.5 B/值 + bf16 scale(2B/32值) = 0.5625  B/值
  改 bf16  = 2.0 B/值
"""
import collections
import json
import struct
import sys

SRC = "/src"


def headers():
    idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    files = sorted(set(idx.values()))
    out = {}
    for f in files:
        with open(f"{SRC}/{f}", "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = v
    return out


def cls(name):
    if ".ffn.experts." in name:
        return "fp4_expert"
    if ("attn." in name or ".indexer." in name or "shared_experts." in name
            or "compressor." in name or ".engram.wkv" in name):
        return "fp8_block"
    return "keep"


def main():
    h = headers()
    n_el = collections.Counter()
    n_t = collections.Counter()
    for k, v in h.items():
        if not k.endswith((".weight", ".scale")):
            continue
        e = 1
        for d in v["shape"]:
            e *= d
        n_el[cls(k)] += e
        n_t[cls(k)] += 1
    tot = sum(n_el.values())
    print(f"{'类别':>10} {'张量数':>8} {'元素数':>16} {'占比':>8} {'源体积 GiB':>12} {'若 bf16 GiB':>12} {'增量 GiB':>10}")
    src_bpv = {"fp4_expert": 0.53125, "fp8_block": 1.0 + 1.0 / 32, "keep": 2.0}
    for c in ("fp4_expert", "fp8_block", "keep"):
        e = n_el[c]
        s = e * src_bpv[c] / 2**30
        b = e * 2.0 / 2**30
        print(f"{c:>10} {n_t[c]:>8} {e:>16,} {e/tot:>8.2%} {s:>12.1f} {b:>12.1f} {b-s:>10.1f}")
    # 方案对比：全面 int4 vs 只把 fp4 专家 int4、fp8 类留 bf16
    planA = n_el["fp4_expert"] * 0.5625 + n_el["fp8_block"] * 0.5625 + n_el["keep"] * 2.0
    planB = n_el["fp4_expert"] * 0.5625 + n_el["fp8_block"] * 2.0 + n_el["keep"] * 2.0
    print(f"\n方案A（全 int4，现状）        = {planA/2**30:.1f} GiB")
    print(f"方案B（仅 fp4 专家 int4，fp8 类留 bf16）= {planB/2**30:.1f} GiB")
    print(f"增量 = {(planB-planA)/2**30:.1f} GiB  ⇒ 每 rank {(planB-planA)/2**30/8:.2f} GiB")
    print(f"\n（参考：现役模型实测 397.3 GiB、每 rank 权重 50.2 GiB、KV 1.97 GiB、卡上限约 62 GiB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
