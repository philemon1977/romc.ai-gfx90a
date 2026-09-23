#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""静态证明：`classify()` 写的格式 与 `ignore` glob 命中，两者必须对每个模块一致。

## 为什么值得单独证一次

起服失败的成本是 ~15 分钟（加载阶段才报错），而报错点离根因很远。不变式很干净：

    classify(M) ∈ {"fp8_to_bf16", "keep"}  ⇒  M 在输出里是**未量化**张量
    ⇒ vLLM 必须把 M 判为 ignore（否则去找 `weight_packed` ⇒ KeyError）
    反之 classify(M) ∈ {"fp4_expert","fp8_block"} ⇒ M 是 int4 packed
    ⇒ vLLM 必须**不**判 ignore（否则去找 `.weight` ⇒ KeyError）

两个方向都会炸，所以要双向都查，而且要覆盖**所有**模块（含 40/41/42 这三个 DSpark 层
——它们模块前缀与主干不同，正是分类器正则最容易漏、glob 最容易不匹配的地方）。

## 融合模块的处理（这是上次 dense-bf16 栽的地方）

vLLM 的 `should_ignore_layer` 拿到的是**模块名**，而 DSV4.1 有三个融合：
    gate_up_proj←[w1,w3]   fused_wqa_wkv←[wq_a,wkv]   fused_wkv_wgate←[wkv,wgate]
且 `find_matching_patterns` 明确"**直接命中融合名优先**"（上一轮已从源码确认）。
⇒ 真实模块名可能是"组件名"也可能是"融合名"，两者我们都试，只要有一个命中就算 ignored
   （若融合名命中，vLLM 走"单一匹配集"分支 ⇒ all()=True ⇒ 与我们的断言一致 ✓）。

权威映射取自**起服时真正加载的那个文件**（我们的补丁），不是印象。
"""
import fnmatch
import json
import re
import sys

sys.path.insert(0, "/w/quark-int8")
import convert_dsv41_ct_int4 as C  # noqa: E402

SRC = "/src"
NEW = "/new"
PATCH_MODEL = "/w/../ai/patches/gfx90a/ct_w4a16_dsv41/dsv41_amd_model.py"

# 与本次转换一致的开关（run_convert.sh: OPT_SCALE/SHARED_BF16/ATTN_BF16/ATTN_MERGED 全开）
C._SHARED_EXPERTS_BF16 = True
C._ATTN_BF16 = True
C._ATTN_MERGED_BF16 = True


def read_fused_mapping():
    """从起服时真正加载的补丁文件里读 packed_modules_mapping。"""
    txt = open("/patch/dsv41_amd_model.py").read()   # 容器内挂载路径（补丁目录，只读）
    m = re.search(r"packed_modules_mapping\s*=\s*\{(.*?)\}", txt, re.S)
    assert m, "没找到 packed_modules_mapping"
    return eval("{" + m.group(1) + "}")


def ignored(layer, patterns, fused):
    """复刻 should_ignore_layer + find_matching_patterns 的语义（含融合名优先）。"""
    def match(name):
        return [p for p in patterns if fnmatch.fnmatchcase(name, p)]
    if match(layer):
        return True
    proj = layer.split(".")[-1]
    for fname, comps in fused.items():
        if proj in comps:
            fused_name = layer[: -len(proj)] + fname
            if match(fused_name):
                return True
    return False


def main():
    fused = read_fused_mapping()
    print("融合映射（取自起服时加载的补丁文件）:", fused)
    cfg = json.load(open(f"{NEW}/config.json"))
    ignore = cfg["quantization_config"]["ignore"]
    gs = cfg["quantization_config"]["config_groups"]
    gsize = [v["weights"].get("group_size") for v in gs.values()]
    print(f"ignore 条数={len(ignore)}  group_size={gsize}")

    sm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    mods = sorted({k[: -len(".weight")] for k in sm if k.endswith(".weight")})
    # 只看会被量化器处理的 Linear 族（norm/embed/head 等不参与 ignore 判定也没关系，
    # 但为稳妥，把 classify 判为 keep 的也一并核对）
    # 非 Linear 的模块**不可能**被量化（CT 的 targets=["Linear"]）⇒ 不进 ignore 也安全。
    # 这条豁免必须显式写出来并打印，否则要么误报、要么靠读者"心里默认"——两者都不可复核。
    # 已用代码证实：amd/dspark.py:99 main_norm=RMSNorm(...)、amd/model.py:694 norm=RMSNorm(...)
    NON_LINEAR = re.compile(r"(^|\.)(norm|main_norm|attn_norm|ffn_norm|kv_norm|q_norm|k_norm|"
                            r"attn_sink|embed|head|lm_head|q_weight|k_weight|bias|bias_vl)$")
    bad = {"should_be_ignored_but_not": [], "ignored_but_packed": [], "exempt_non_linear": []}
    counts = {}
    for m in mods:
        k = C.classify(m)
        counts[k] = counts.get(k, 0) + 1
        if NON_LINEAR.search(m):
            bad["exempt_non_linear"].append(m)
            continue
        unquant_out = k in ("fp8_to_bf16", "keep")
        ig = ignored(m, ignore, fused)
        if unquant_out and not ig:
            bad["should_be_ignored_but_not"].append((m, k))
        if (not unquant_out) and ig:
            bad["ignored_but_packed"].append((m, k))
    print("分类计数:", counts)
    print(f"\n非 Linear 豁免（已用代码证实不会被量化）{len(bad['exempt_non_linear'])} 个: "
          f"{sorted(set(x.split('.')[-1] for x in bad['exempt_non_linear']))}")
    for name, lst in bad.items():
        if name == "exempt_non_linear":
            continue
        print(f"\n{name}: {len(lst)} 个")
        for m, k in lst[:12]:
            print(f"   {m}   [classify={k}]")

    # DSpark 层单独列出来（模块前缀与主干不同，最易出问题）
    dsp = sorted({m for m in mods if re.match(r"^(layers\.(4[0-2])\.|dspark\.|mtp\.)", m)})
    print(f"\nDSpark/MTP 相关模块数={len(dsp)}  样本: {dsp[:6]}")
    badD = [m for m in dsp if not NON_LINEAR.search(m)
            and (C.classify(m) in ("fp8_to_bf16", "keep")) != ignored(m, ignore, fused)]
    print(f"DSpark 不变式违反数 = {len(badD)}  {badD[:6]}")
    ok = not bad["should_be_ignored_but_not"] and not bad["ignored_but_packed"] and not badD
    print(f"\n=== 不变式全局成立: {'✅ 是' if ok else '❌ 否'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
