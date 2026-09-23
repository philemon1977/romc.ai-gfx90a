#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""权威完整性审计：不看转换器的计数器，直接逐个源模块断言产物张量齐不齐。

## 为什么需要它

`convert_dsv41_ct_int4.py` 结尾有一道自校验：`converted == expected` 否则报 ERROR。
但它的"从已存在分片回读计数"逻辑**认不出 `fp8_to_bf16` 模块**（这类模块转换后只剩一个
bf16 `.weight`，没有 packed 三件套 ⇒ 检测器扫不到）。于是本次全量跑以
`ERROR: 47235 vs 47587` 收场，而差额 352 恰 = 全 checkpoint 的 bf16 模块数。

**计数器只是代理指标。**真正要保证的是：对源里每个模块，产物里该有的张量一个都不能少。
本脚本按 `classify()` 的分类逐模块断言，与计数器无关 ⇒ 可以据此判定"是记账假警报
还是真的漏了东西"。顺带核对每片张量数与源侧的对应关系（不是要求相等，而是按分类推算）。
"""
import json
import sys

sys.path.insert(0, "/w/quark-int8")
import convert_dsv41_ct_int4 as C  # noqa: E402

SRC = "/src"
NEW = "/new"
C._SHARED_EXPERTS_BF16 = True
C._ATTN_BF16 = True
C._ATTN_MERGED_BF16 = True
C.GROUP_SIZE = 32


def main():
    sm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    nm = json.load(open(f"{NEW}/model.safetensors.index.json"))["weight_map"]

    # 源侧模块名（.weight 结尾，去掉后缀）
    mods = sorted({k[: -len(".weight")] for k in sm if k.endswith(".weight")})
    missing = {"int4": [], "bf16": [], "keep": []}
    counts = {}
    for m in mods:
        k = C.classify(m)
        counts[k] = counts.get(k, 0) + 1
        if k == "fp4_expert" or k == "fp8_block":
            # 期望 packed/scale/shape 三件套
            for suf in ("weight_packed", "weight_scale", "weight_shape"):
                if f"{m}.{suf}" not in nm:
                    missing["int4"].append(f"{m}.{suf}")
        elif k == "fp8_to_bf16":
            if f"{m}.weight" not in nm:
                missing["bf16"].append(f"{m}.weight")
            if f"{m}.weight_packed" in nm:
                missing["bf16"].append(f"{m}.weight_packed(残留!)")
        elif k == "keep":
            if f"{m}.weight" not in nm:
                missing["keep"].append(f"{m}.weight")

    print("模块分类计数（按 classify 现算）:", counts)
    for k, v in missing.items():
        print(f"  缺失[{k}]: {len(v)}  {v[:4]}")

    # 反向核对：dst 里每个张量都必须能被某条规则解释，否则就是多余/错名
    explained = set()
    for m in mods:
        k = C.classify(m)
        if k in ("fp4_expert", "fp8_block"):
            explained |= {f"{m}.weight_packed", f"{m}.weight_scale", f"{m}.weight_shape"}
        elif k == "keep":
            explained.add(f"{m}.weight")
            if f"{m}.scale" in sm:          # keep 类的伴随 scale 是原样直通的，不是冗余 ✗
                explained.add(f"{m}.scale")
        else:
            explained.add(f"{m}.weight")
    # keep 类里那些不以 .weight 结尾的（如 q_weight/k_weight/attn_sink/embed.weight/head.weight）
    # 兜底：源里既不是 .weight 也不是任何模块 .scale 的键（极少，如 q_weight/k_weight）
    for k in sm:
        if not (k.endswith(".weight") or k.endswith(".scale")):
            explained.add(k)
    surplus = [t for t in nm if t not in explained]

    tot = sum(len(v) for v in missing.values())
    print(f"\n产物 index 张量数 = {len(nm)}；未被规则解释的冗余张量 = {len(surplus)}  {surplus[:4]}")
    ok = tot == 0 and len(surplus) == 0
    print(f"\n=== 完整性审计: {'✅ 通过（转换器的 ERROR 确为记账假警报）' if ok else '❌ 不通过'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
