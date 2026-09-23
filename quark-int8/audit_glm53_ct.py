# -*- coding: utf-8 -*-
"""GLM-5.3 CT-Int4 仓完整性审计（不依赖转换器内部函数，独立判据）。

检查：
 1) 索引完整：源里每个被判定量化的模块，产物必须有 weight_packed/weight_scale/weight_shape
 2) 非量化模块：产物保留原始 .weight（scale 若为 fp8 源则保留/F32 源则丢弃均可，但不得缺失 weight）
 3) ignore glob 与量化集合**不相交**（否则 vLLM 会把量化模块当未量化构建 ⇒ 装载失败）
 4) config.json 的量化字段与目录/分片数自洽
用法: python3 audit_glm53_ct.py <源目录> <产物目录>
"""
import fnmatch, json, os, re, struct, sys, collections

SRC, OUT = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")

def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

def index_of(d):
    return json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]

si, oi = index_of(SRC), index_of(OUT)
print("[1] 张量数: 源 %d  产物 %d" % (len(si), len(oi)))

def classify(mod):
    # ★ 必须与 convert_glm53_ct_int4.py::classify 同步（dense MLP 规则漏过一次，
    #   导致审计误报 9 个"缺失/多余"）
    if re.search(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj$", mod):
        return "int4"
    if re.search(r"\.mlp\.(gate|up|down)_proj$", mod):
        return "int4"
    if re.search(r"\.mlp\.shared_experts\.(gate|up|down)_proj$", mod):
        return "bf16"          # 默认共享专家 bf16
    if re.search(r"\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$", mod):
        return "int4"
    # ★ indexer.wk 在 vLLM 里与 weights_proj 融合成 wk_weights_proj 一个模块，
    #   只能有一种 scheme ⇒ 与转换器一致：保持未量化（源 fp8 → 反量化成 bf16）。
    if re.search(r"\.self_attn\.indexer\.wk$", mod):
        return "bf16"
    if re.search(r"\.self_attn\.indexer\.wq_b$", mod):
        return "int4"
    return "keep"

want_q, want_k, missing = [], [], []
for k in si:
    if not k.endswith(".weight"):
        continue
    mod = k[: -len(".weight")]
    c = classify(mod)
    if c == "int4":
        want_q.append(mod)
        for suf in (".weight_packed", ".weight_scale", ".weight_shape"):
            if mod + suf not in oi:
                missing.append(mod + suf)
    else:
        want_k.append(mod)
        if k not in oi:
            missing.append(k)
print("[2] 应量化模块 %d  应保留 weight %d" % (len(want_q), len(want_k)))
extra = [k for k in oi if k.endswith(".weight_packed") and k[: -len(".weight_packed")] not in set(want_q)]
print("[3] 缺失张量: %d %s" % (len(missing), missing[:5]))
print("[4] 多余 weight_packed: %d %s" % (len(extra), extra[:5]))

cfg = json.load(open(os.path.join(OUT, "config.json")))
ign = cfg["quantization_config"]["ignore"]
# ★ 用 vLLM 自己的判定函数（与 ① 补丁同一调用：use_fnmatch=True），
#   而不是自造的 fnmatch 近似——2026-09-20 实测：自造匹配把 "*mlp.gate" 当成也能命中
#   mlp.gate_proj，产生 3 条误报。
try:
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer as _sil,
    )
    _is_ign = lambda m: bool(_sil(m, ignore=ign, use_fnmatch=True))  # noqa: E731
    _src = "vLLM should_ignore_layer"
except Exception as _e:  # pragma: no cover
    _is_ign = lambda m: any(fnmatch.fnmatch(m, p) for p in ign)  # noqa: E731
    _src = "fnmatch 兜底 (%s)" % _e
hits = [m for m in want_q if _is_ign(m)]
print("[5] 量化模块被 ignore 命中的数量: %d %s   (判据: %s)" % (len(hits), hits[:5], _src))

# [4b] ★ 真正的判据：产物里每个"保留的"线性层 .weight 必须被 ignore 命中，
#      否则 vLLM 会把它当量化层建 ⇒ KeyError（这正是 dense MLP 那次起服失败的形态）。
kept_mods = sorted({k[: -len(".weight")] for k in oi
                    if k.endswith(".weight") and not k.endswith(".weight_scale_inv")})
not_ignored = []
for mod in kept_mods:
    short = mod[len("model."):] if mod.startswith("model.") else mod
    if not any(fnmatch.fnmatch(mod, p) or fnmatch.fnmatch(short, p) or p in mod or p in short
               for p in ign):
        not_ignored.append(mod)
print("[4b] 保留但未被 ignore 命中的模块: %d %s" % (len(not_ignored), not_ignored[:6]))

# [4c] 被 ignore 命中的模块若在**源**里是 fp8（带 weight_scale_inv），则 vLLM 会按未量化(bf16)构建
#      却拿到 fp8 字节 ⇒ 静默错。这类模块必须在转换时 dequant 成 bf16。
def _prod_dtype(mod):
    k = mod + ".weight"
    if k not in oi:
        return None
    with open(os.path.join(OUT, oi[k]), "rb") as _f:
        _n = struct.unpack("<Q", _f.read(8))[0]
        _h = json.loads(_f.read(_n))
    return _h[k]["dtype"]

fp8_kept = []
for mod in kept_mods:
    if not any(fnmatch.fnmatch(mod, p) or p in mod for p in ign):
        continue
    # ★ 判据在**产物**上：被 ignore 的模块，其 weight 必须是 bf16（不能还是 fp8 字节）
    if _prod_dtype(mod) in ("F8_E4M3", "F8_E5M2", "U8"):
        fp8_kept.append(mod)
print("[4c] 被 ignore 但源为 fp8 的模块: %d %s" % (len(fp8_kept), fp8_kept[:6]))


shards = sorted(set(oi.values()))
print("[6] 分片 %d 个" % len(shards))
tot = sum(os.path.getsize(os.path.join(OUT, f)) for f in shards)
print("[7] 产物大小 %.1f GiB" % (tot / 2**30))
qc = cfg["quantization_config"]
print("[8] 量化配置:", json.dumps({k: qc[k] for k in ("quant_method", "format")}, ensure_ascii=False),
      " group_size=", qc["config_groups"]["group_0"]["weights"]["group_size"],
      " bits=", qc["config_groups"]["group_0"]["weights"]["num_bits"])
# [9] 漂移检测：审计器自带的 classify 必须与转换器的 classify **逐模块一致**，
#     否则本审计的所有判据都建立在过时策略上（2026-09-20 真踩过：转换器加了 dense MLP 规则，
#     审计器没同步，凭空报出 9 个"缺失/多余"）。判据可以有两份，但必须互相校验。
drift = []
try:
    import importlib.util
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "convert_glm53_ct_int4.py")
    _sp = importlib.util.spec_from_file_location("_glm53_converter", _p)
    _cv = importlib.util.module_from_spec(_sp)
    _sp.loader.exec_module(_cv)
    for k in si:
        if not k.endswith(".weight"):
            continue
        m = k[: -len(".weight")]
        want = _cv.classify(m)
        got = classify(m)
        same = ((want == "fp4_expert" or want == "fp8_block") and got == "int4") or \
               (want == "fp8_to_bf16" and got == "bf16") or (want == "keep" and got == "keep")
        if not same:
            drift.append((m, want, got))
    print("[9] classify 与转换器漂移: %d %s" % (len(drift), drift[:4]))
except SystemExit:
    raise
except Exception as _e:  # noqa: BLE001
    drift.append(("导入失败", str(_e)[:80], ""))
    print("[9] ⚠ 无法加载转换器做漂移校验: %r" % _e)

ok = ((not missing) and (not extra) and (not hits) and (not not_ignored)
      and (not fp8_kept) and (not drift))
print("\nAUDIT:", "PASS ✅" if ok else "FAIL ❌")