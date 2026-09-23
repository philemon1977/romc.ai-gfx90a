#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按分片断言"量化策略是否真的按开关生效"——起服前的结构性 QA。

## 为什么值得做

`-opt-allbf16` 这个 checkpoint 承载了四个开关（最优 scale / 共享专家 bf16 /
注意力 bf16 / 合并模块 bf16）。如果某个分片没生效（例如分类器的正则没覆盖到某层的
特殊命名），**要等 15 分钟起服、在加载阶段才炸**，而且报错点与根因隔得很远。
本脚本用几十秒把每个已完成分片过一遍，把这类问题挡在起服之前。

## 断言（对每层的注意力 + MoE）

  bf16（应存在 `<mod>.weight`，且**不得**出现 `.weight_packed`）：
    attn.wq_a, attn.wkv, attn.wq_b, attn.wo_a, attn.wo_b, attn.indexer.wq_b,
    ffn.shared_experts.{w1,w2,w3}
  int4（应存在 packed/scale/shape 三件套）：
    ffn.experts.*.{w1,w2,w3}
  并抽样核对一个 bf16 张量与源 fp8 的偏差应 ≈ 0（无损）。
"""
import json
import os
import struct
import sys

import torch
from safetensors import safe_open

SRC = "/src"
NEW = "/new"
# ★ 这个清单必须与 config 的 quantization_config.ignore 里"新增的 bf16 条目"**一一对应**：
#   任何一侧多出来都会炸——ignore 命中而张量仍是 int4 ⇒ vLLM 找 .weight 得 KeyError；
#   反之（转成 bf16 却没 ignore）⇒ vLLM 找 weight_packed 同样 KeyError。
#   漏掉 indexer.wq_b 曾是本脚本的真实空洞（它只在 8 个 index-source 层存在，故必须"有则断言"）。
BF16_EXPECT = ["attn.wq_a", "attn.wkv", "attn.wq_b", "attn.wo_a", "attn.wo_b",
               "attn.indexer.wq_b",
               "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3"]


def main():
    shards = sys.argv[1:]
    # ★ 不能读输出目录的 index：转换未完时它仍是试点那份（只含分片 3）⇒ 会把好分片误判成缺失。
    #   所以新模型一侧**直接读每个分片自己的 header**；源一侧用完整的源 index ✓。
    def new_names_of(shard_file):
        with open(shard_file, "rb") as fh:
            ln = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(ln))
        return {k for k in h if k != "__metadata__"}
    nm = {}          # 占位：下面一律用 have（集合）判断
    sm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    bad = 0
    for sh in shards:
        path = f"{NEW}/{sh}"
        if not os.path.exists(path):
            print(f"⏳ {sh}: 文件还没写出（跳过，非错误）")
            continue
        have = new_names_of(path)
        names = sorted(have)
        if not names:
            print(f"{sh}: ❌ header 里没有张量")
            bad += 1
            continue
        layers = sorted({int(k.split(".")[1]) for k in names if k.startswith("layers.")})
        errs = []
        n_experts = 0
        for L in layers:
            for m in BF16_EXPECT:
                base = f"layers.{L}.{m}"
                # 有的层没有某些模块（indexer 只在源层）⇒ 只在源里存在时才断言
                src_has = f"{base}.weight" in sm
                if not src_has:
                    continue
                if base + ".weight" not in have:
                    errs.append(f"L{L} {m}: 缺 bf16 weight")
                if base + ".weight_packed" in have:
                    errs.append(f"L{L} {m}: 仍残留 weight_packed ✗")
            for w in ("w1", "w2", "w3"):
                base = f"layers.{L}.ffn.experts.0.{w}"
                if f"{base}.weight" in sm and base + ".weight_packed" not in have:
                    errs.append(f"L{L} experts.0.{w}: 缺 int4 packed")
            n_experts = sum(1 for k in names if f"layers.{L}.ffn.experts." in k and k.endswith("w1.weight_packed"))
        with safe_open(path, framework="pt") as f:
            dt_bad = []
            for L in layers[:1]:
                for m in ("attn.wq_a", "attn.wq_b", "ffn.shared_experts.w1"):
                    base = f"layers.{L}.{m}"
                    if base + ".weight" in have:
                        t = f.get_tensor(f"{base}.weight")
                        if t.dtype != torch.bfloat16:
                            dt_bad.append(f"{base}:{t.dtype}")
        # 抽样无损核对（每片一次）
        rel_txt = ""
        if layers:
            mod = f"layers.{layers[0]}.ffn.shared_experts.w1"
            if mod + ".weight" in have and f"{mod}.weight" in sm:
                with safe_open(path, framework="pt") as f:
                    got = f.get_tensor(mod + ".weight").float()
                with safe_open(f"{SRC}/{sm[mod + '.weight']}", framework="pt") as g:
                    w0 = g.get_tensor(mod + ".weight")
                    s0 = g.get_tensor(mod + ".scale")
                sys.path.insert(0, "/w/quark-int8")
                from convert_dsv41_ct_int4 import dequant_fp8_block
                ref = dequant_fp8_block(w0, s0, dtype=torch.float32)
                rel = ((got.double().flatten() - ref.double().flatten()).norm()
                       / ref.double().flatten().norm()).item()
                rel_txt = f" 共享专家 vs 源 rel={rel:.4%}"
                if rel > 1e-6:
                    errs.append(f"共享专家 rel={rel:.4%} 非无损 ✗")
        status = "✅" if not errs and not dt_bad else "❌"
        if errs or dt_bad:
            bad += 1
        print(f"{status} {sh} 层={layers} 专家片={n_experts}{rel_txt}"
              + (f"  问题: {errs[:3]}{dt_bad[:2]}" if (errs or dt_bad) else ""), flush=True)
    print(f"\n=== 不合格分片数 = {bad} ===")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
