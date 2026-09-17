#!/usr/bin/env python3
"""Verify a converted compressed-tensors INT4 checkpoint against its source.

Structural checks run on **every** converted module; numeric checks run on a
random sample (``--sample``, default 24) because unpacking 1152 experts per
shard costs minutes of CPU for no extra information.

Checks:
  * output tensor-name set == expected set (converted -> packed/scale/shape)
  * dtypes/shapes conform to pack-quantized conventions
  * weight_shape matches the logical source shape (packed_cols*2 for FP4 experts)
  * sampled: unpack(int4)*scale reproduces the source-dequantized weight
"""
import argparse
import json
import os
import random
import re
import sys

import torch
from safetensors import safe_open

GROUP = 32
FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e8m0_to_f32(scale):
    u = scale.view(torch.uint8).to(torch.int32) - 127
    return torch.exp2(u.float())


def dequant_fp4_expert(packed, scale):
    b = packed.contiguous().view(torch.uint8)
    hi = (b >> 4) & 0x0F
    lo = b & 0x0F
    q = torch.stack((hi, lo), dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)
    lut = torch.tensor(FP4_LUT, dtype=torch.float32)
    val = lut[(q & 0x07).long()]
    val = torch.where((q & 0x08) != 0, -val, val)
    return val * e8m0_to_f32(scale).repeat_interleave(32, dim=1)


def dequant_fp8_block(w, scale):
    n, k = w.shape
    br, bc = n // scale.shape[0], k // scale.shape[1]
    s = e8m0_to_f32(scale).repeat_interleave(br, dim=0).repeat_interleave(bc, dim=1)
    return w.to(torch.float32) * s


def unpack_int4(packed, scale):
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    q = ((packed.unsqueeze(-1) >> sh) & 0xF).reshape(packed.shape[0], -1).to(torch.float32) - 8
    return q * scale.repeat_interleave(GROUP, dim=1)


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
    if module.endswith(".engram.wkv") or module.endswith(".engram.embed"):
        return "fp8_block"
    return "keep"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("shard")
    ap.add_argument("--sample", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    swm = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    owm = json.load(open(os.path.join(a.out, "model.safetensors.index.json")))["weight_map"]
    s_names = {n for n, s in swm.items() if s == a.shard}
    o_names = {n for n, s in owm.items() if s == a.shard}
    print(f"shard {a.shard}: source {len(s_names)} -> output {len(o_names)} tensors")

    exp, conv = set(), []
    for n in s_names:
        mod = n[: -len(".weight")] if n.endswith(".weight") else (
            n[: -len(".scale")] if n.endswith(".scale") else None)
        if mod is not None and classify(mod) != "keep":
            # consumed by conversion; replaced by packed/scale/shape
            if n.endswith(".weight"):
                exp |= {f"{mod}.weight_packed", f"{mod}.weight_scale", f"{mod}.weight_shape"}
                conv.append((mod, classify(mod)))
            continue
        exp.add(n)
    missing, extra = exp - o_names, o_names - exp
    print(f"  converted modules: {len(conv)}")
    print(f"  name set: missing={len(missing)} extra={len(extra)}")
    for n in sorted(missing)[:5]:
        print("    MISSING", n)
    for n in sorted(extra)[:5]:
        print("    EXTRA  ", n)

    fails = []
    rng = random.Random(a.seed)
    sample = set(rng.sample([m for m, _ in conv], min(a.sample, len(conv))))
    with safe_open(os.path.join(a.out, a.shard), framework="pt") as fo, \
         safe_open(os.path.join(a.src, a.shard), framework="pt") as fs:
        for mod, kind in conv:
            pk = fo.get_slice(f"{mod}.weight_packed")
            sc = fo.get_slice(f"{mod}.weight_scale")
            shp = fo.get_tensor(f"{mod}.weight_shape")
            ws = fs.get_slice(f"{mod}.weight")
            n, kb = ws.get_shape()
            k_log = kb * 2 if kind == "fp4_expert" else kb
            if tuple(shp.tolist()) != (n, k_log):
                fails.append(f"{mod}: weight_shape {tuple(shp.tolist())} != {(n, k_log)}")
                continue
            if str(pk.get_dtype()) != "I32" or tuple(pk.get_shape()) != (n, k_log // 8):
                fails.append(f"{mod}: packed {pk.get_dtype()} {tuple(pk.get_shape())} != I32 {(n, k_log//8)}")
                continue
            if str(sc.get_dtype()) != "F32" or tuple(sc.get_shape()) != (n, k_log // GROUP):
                fails.append(f"{mod}: scale {sc.get_dtype()} {tuple(sc.get_shape())} != F32 {(n, k_log//GROUP)}")
                continue
            if k_log % GROUP:
                fails.append(f"{mod}: K={k_log} % {GROUP} != 0")
                continue

            if mod not in sample:
                continue
            ref = (dequant_fp4_expert(fs.get_tensor(f"{mod}.weight"), fs.get_tensor(f"{mod}.scale"))
                   if kind == "fp4_expert" else
                   dequant_fp8_block(fs.get_tensor(f"{mod}.weight"), fs.get_tensor(f"{mod}.scale")))
            rec = unpack_int4(fo.get_tensor(f"{mod}.weight_packed"), fo.get_tensor(f"{mod}.weight_scale"))
            rel = float((rec - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
            cos = float(torch.nn.functional.cosine_similarity(rec.flatten(), ref.flatten(), dim=0))
            ok = rel < 0.15 and cos > 0.99
            print(f"    {'ok ' if ok else 'BAD'} {kind:10s} {mod[:60]:60s} relRMSE={rel:.4f} cos={cos:.4f} shape={tuple(rec.shape)}")
            if not ok:
                fails.append(f"{mod}: relRMSE={rel:.4f} cos={cos:.4f}")

    print(f"  structural checks: {len(conv)} modules; numeric sample: {len(sample)}")
    print(f"  FAILURES: {len(fails)}")
    for f_ in fails[:10]:
        print("   ", f_)
    ss = os.path.getsize(os.path.join(a.src, a.shard))
    so = os.path.getsize(os.path.join(a.out, a.shard))
    print(f"  size: {ss/2**30:.3f} GiB -> {so/2**30:.3f} GiB ({100*(so-ss)/ss:+.1f}%)")
    return 1 if (fails or missing or extra) else 0


if __name__ == "__main__":
    sys.exit(main())
