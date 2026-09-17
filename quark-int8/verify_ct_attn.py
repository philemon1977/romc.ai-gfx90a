#!/usr/bin/env python3
"""Targeted numeric check of the fp8_block -> int4 attention conversions.

The random sampler in verify_ct_int4.py is dominated by experts (47k experts vs
353 attention modules), so attention needs an explicit check.
"""
import json
import os
import sys

import torch
from safetensors import safe_open

GROUP = 32
E8M0 = torch.float8_e8m0fnu


def e8m0_to_f32(s):
    return torch.exp2((s.view(torch.uint8).to(torch.int32) - 127).float())


def dequant_fp8_block(w, s):
    n, k = w.shape
    br, bc = n // s.shape[0], k // s.shape[1]
    sc = e8m0_to_f32(s).repeat_interleave(br, 0).repeat_interleave(bc, 1)
    return w.to(torch.float32) * sc


def unpack_int4(p, s):
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    q = ((p.unsqueeze(-1) >> sh) & 0xF).reshape(p.shape[0], -1).to(torch.float32) - 8
    return q * s.repeat_interleave(GROUP, 1)


def main():
    src, out = sys.argv[1], sys.argv[2]
    mods = sys.argv[3:]
    swm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    owm = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
    fails = 0
    for mod in mods:
        sf, of = swm[f"{mod}.weight"], owm[f"{mod}.weight_packed"]
        with safe_open(os.path.join(src, sf), "pt") as f:
            w, s = f.get_tensor(f"{mod}.weight"), f.get_tensor(f"{mod}.scale")
        with safe_open(os.path.join(out, of), "pt") as f:
            pk, sc = f.get_tensor(f"{mod}.weight_packed"), f.get_tensor(f"{mod}.weight_scale")
            shp = f.get_tensor(f"{mod}.weight_shape")
        ref = dequant_fp8_block(w, s)
        rec = unpack_int4(pk, sc)
        rel = float((rec - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
        cos = float(torch.nn.functional.cosine_similarity(rec.flatten(), ref.flatten(), dim=0))
        shape_ok = tuple(shp.tolist()) == (w.shape[0], w.shape[1])
        ok = rel < 0.15 and cos > 0.99 and shape_ok and rec.shape == ref.shape
        fails += 0 if ok else 1
        print(f"  {'ok ' if ok else 'BAD'} {mod:42s} src{tuple(w.shape)}->int4{tuple(rec.shape)} "
              f"scale{tuple(sc.shape)} shape_ok={shape_ok} relRMSE={rel:.4f} cos={cos:.4f}")
    print(f"  FAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
