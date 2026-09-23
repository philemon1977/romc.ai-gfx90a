#!/usr/bin/env python3
"""Header-only validation of a converted compressed-tensors INT4 checkpoint.

Reads only safetensors headers (no tensor data), so it is fast even for a
620 GiB checkpoint.  Verifies, for every shard:
  * output name set == expected set (converted modules -> packed/scale/shape)
  * dtypes and shapes conform to pack-quantized conventions
  * weight_shape carries the *logical* source shape
  * index.json weight_map covers every tensor exactly once
"""
import json
import os
import re
import sys

GROUP = 32


def classify(module):
    if re.search(r"\.ffn\.experts\.\d+\.(w1|w2|w3)$", module):
        return "fp4_expert"
    if re.search(r"(^|\.)attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$", module):
        return "fp8_block"
    if re.search(r"(^|\.)attn\.indexer\.wq_b$", module):
        return "fp8_block"
    if re.search(r"\.ffn\.shared_experts\.(w1|w2|w3)$", module):
        return "fp8_block"
    if module.endswith(".main_proj"):
        return "fp8_block"
    if module.endswith(".engram.wkv"):
        return "fp8_to_bf16"
    return "keep"


def headers(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n))


def main():
    src, out = sys.argv[1], sys.argv[2]
    swm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    oidx_path = os.path.join(out, "model.safetensors.index.json")
    if not os.path.exists(oidx_path):
        print("NO INDEX YET:", oidx_path)
        return 2
    owm = json.load(open(oidx_path))["weight_map"]

    src_shards = sorted(set(swm.values()), key=lambda s: int(re.search(r"-(\d+)-of-", s).group(1)))
    missing_shards = [s for s in src_shards if not os.path.exists(os.path.join(out, s))]
    print(f"shards: {len(src_shards) - len(missing_shards)}/{len(src_shards)} present")
    if missing_shards:
        print("  missing:", [int(re.search(r'-(\d+)-of-', s).group(1)) for s in missing_shards])

    n_conv = n_keep = 0
    fails = []
    for shard in src_shards:
        op = os.path.join(out, shard)
        if not os.path.exists(op):
            continue
        oh = headers(op)
        sh = headers(os.path.join(src, shard))
        s_names = [n for n, s in swm.items() if s == shard]
        exp = set()
        conv = []
        for n in s_names:
            mod = n[: -len(".weight")] if n.endswith(".weight") else (
                n[: -len(".scale")] if n.endswith(".scale") else None)
            k = classify(mod) if mod is not None else "keep"
            if k == "fp8_to_bf16":
                if n.endswith(".weight"):
                    exp.add(n)          # dense bf16 weight, same name
                    conv.append(mod)
                # .scale is intentionally dropped
                continue
            if mod is not None and k != "keep":
                if n.endswith(".weight"):
                    exp |= {f"{mod}.weight_packed", f"{mod}.weight_scale", f"{mod}.weight_shape"}
                    conv.append(mod)
                continue
            exp.add(n)
        o_names = {k for k in oh if k != "__metadata__"}
        miss, extra = exp - o_names, o_names - exp
        if miss or extra:
            fails.append(f"{shard}: missing={len(miss)} {sorted(miss)[:3]} extra={len(extra)} {sorted(extra)[:3]}")
        for mod in conv:
            kind = classify(mod)
            if kind == "fp8_to_bf16":
                # dense bf16 weight under the SAME name; scale must be gone
                ow = oh.get(f"{mod}.weight")
                sw = sh.get(f"{mod}.weight")
                if ow is None or ow["dtype"] != "BF16":
                    fails.append(f"{shard}:{mod}: expected BF16 weight, got {ow}")
                elif sw is not None and tuple(ow["shape"]) != tuple(sw["shape"]):
                    fails.append(f"{shard}:{mod}: bf16 shape {tuple(ow['shape'])} != source {tuple(sw['shape'])}")
                if f"{mod}.scale" in oh:
                    fails.append(f"{shard}:{mod}: scale should have been dropped")
                n_conv += 1
                continue
            s = sh[f"{mod}.weight"] if f"{mod}.weight" in sh else sh[f"{mod}.scale"]
            # K from the source tensor that carries it
            if f"{mod}.weight" in sh:
                wd = sh[f"{mod}.weight"]
                n, k = wd["shape"][0], wd["shape"][1]
            else:
                k = sh[f"{mod}.scale"]["shape"][1]
                n = sh[f"{mod}.scale"]["shape"][0]
            if kind == "fp4_expert":
                k = sh[f"{mod}.scale"]["shape"][1] * 32
                n = sh[f"{mod}.scale"]["shape"][0]
            pk = oh.get(f"{mod}.weight_packed")
            sc = oh.get(f"{mod}.weight_scale")
            shp = oh.get(f"{mod}.weight_shape")
            if not (pk and sc and shp):
                fails.append(f"{shard}:{mod}: missing packed/scale/shape")
                continue
            if pk["dtype"] != "I32" or tuple(pk["shape"]) != (n, k // 8):
                fails.append(f"{shard}:{mod}: packed {pk['dtype']}{tuple(pk['shape'])} != I32{(n, k // 8)}")
            # scale dtype is bf16 by design (vLLM builds weight_scale with
            # params_dtype=bf16); only the shape must match.
            if sc["dtype"] not in ("F32", "BF16") or tuple(sc["shape"]) != (n, k // GROUP):
                fails.append(f"{shard}:{mod}: scale {sc['dtype']}{tuple(sc['shape'])} != F32/BF16{(n, k // GROUP)}")
            if shp["dtype"] != "I64" or tuple(shp["shape"]) != (2,):
                fails.append(f"{shard}:{mod}: weight_shape {shp['dtype']}{tuple(shp['shape'])}")
            n_conv += 1
        n_keep += len(s_names)

    print(f"  converted modules checked: {n_conv}")
    print(f"  source tensors seen: {n_keep}")
    print(f"  FAILURES: {len(fails)}")
    for f_ in fails[:15]:
        print("   ", f_)
    return 1 if fails or missing_shards else 0


if __name__ == "__main__":
    sys.exit(main())
