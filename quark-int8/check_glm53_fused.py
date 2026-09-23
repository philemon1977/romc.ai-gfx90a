# -*- coding: utf-8 -*-
"""融合模块一致性检查（离线，CPU 只读）——把"融合模块只能有一种 scheme"这条教训自动化。

vLLM 的融合组（证据：GlmMoeDsaForCausalLM.packed_modules_mapping + 源码 __init__ + indexer 实际行为）：
  gate_up_proj      = [gate_proj, up_proj]                     （dense MLP / 共享专家）
  fused_qkv_a_proj  = [q_a_proj, kv_a_proj_with_mqa]           （q_lora_rank 存在时）
  wk_weights_proj   = [wk, weights_proj]                       （DSA indexer；上次炸装载就是它）
  专家 w13           = [gate_proj, up_proj]（逐专家，get_expert_mapping）
检查：同一融合组内所有分片必须同为"量化"或同为"未量化"，且不得缺片。
"""
import json, re, sys, collections

O = sys.argv[1].rstrip("/")
oi = json.load(open(O + "/model.safetensors.index.json"))["weight_map"]

def has(mod, suf):
    return (mod + "." + suf) in oi

def scheme(mod):
    if has(mod, "weight_packed"):
        return "int4"
    if has(mod, "weight"):
        return "bf16/keep"
    return "缺失"

GROUPS = [
    ("gate_up_proj",     ["mlp.gate_proj", "mlp.up_proj"]),
    ("fused_qkv_a_proj", ["self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"]),
    ("wk_weights_proj",  ["self_attn.indexer.wk", "self_attn.indexer.weights_proj"]),
    ("shared_experts",   ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"]),
]
layers = sorted({int(m.group(1)) for k in oi if (m := re.match(r"model\.layers\.(\d+)\.", k))})
print("层数: %d (%d..%d)" % (len(layers), layers[0], layers[-1]))
bad = 0
for gname, shards in GROUPS:
    stats = collections.Counter()
    for L in layers:
        mods = ["model.layers.%d.%s" % (L, s) for s in shards]
        if not any(has(m, "weight") or has(m, "weight_packed") for m in mods):
            continue
        schemes = [scheme(m) for m in mods]
        stats[tuple(schemes)] += 1
        if len(set(schemes)) != 1 or "缺失" in schemes:
            bad += 1
            if bad <= 5:
                print("  ✗ 层 %d %s: %s" % (L, gname, list(zip(shards, schemes))))
    print("  %-18s %s" % (gname, dict(stats)))
print()
print("=== 专家 w13 组（逐专家 gate/up 必须同 scheme，抽样）===")
exp_bad = 0
sample = [k for k in oi if re.search(r"\.mlp\.experts\.\d+\.gate_proj\.weight_packed$", k)][:3]
for k in sample:
    base = k[: -len("gate_proj.weight_packed")]
    g, u, d = scheme(base + "gate_proj"), scheme(base + "up_proj"), scheme(base + "down_proj")
    ok = (g == u == d == "int4")
    exp_bad += (not ok)
    print("  %-52s gate=%s up=%s down=%s %s" % (base[-52:], g, u, d, "OK" if ok else "MISMATCH"))
print()
print("组内不一致/缺片: %d ; 专家抽样不一致: %d" % (bad, exp_bad))
print("FUSED_CHECK:", "PASS" if (bad == 0 and exp_bad == 0) else "FAIL")
