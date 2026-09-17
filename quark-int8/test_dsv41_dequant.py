#!/usr/bin/env python3
"""Unit-test the DSV4.1 dequantizers/quantizer on real checkpoint tensors."""
import importlib.util
import json
import os

import torch
from safetensors import safe_open

spec = importlib.util.spec_from_file_location("cv", "/w/quark-int8/convert_dsv41_ct_int4.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

S = "/mnt/stripe-3mix-3t2/models/deepseek-ai/DeepSeek-V4.1-Flash"
wm = json.load(open(S + "/model.safetensors.index.json"))["weight_map"]
dev = torch.device("cuda")


def get(n):
    with safe_open(os.path.join(S, wm[n]), "pt") as f:
        return f.get_tensor(n)


def unpack(packed, scale, N, K, gs=32):
    sh = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    q = ((packed.unsqueeze(-1) >> sh) & 0xF).reshape(N, K).to(torch.float32) - 8
    return q * scale.repeat_interleave(gs, dim=1)


print("=== 1) int4 g32 quantizer round-trip (synthetic) ===")
torch.manual_seed(0)
w = torch.randn(256, 512, dtype=torch.bfloat16, device=dev)
p = m.pack_int4_g32(w, 32)
rec = unpack(p["weight_packed"], p["weight_scale"], 256, 512)
rel = (rec - w.float()).pow(2).mean().sqrt() / w.float().pow(2).mean().sqrt()
cos = torch.nn.functional.cosine_similarity(rec.flatten(), w.float().flatten(), dim=0)
print(f"  relRMSE={float(rel):.5f} cos={float(cos):.5f}  (expect relRMSE ~0.12, cos ~0.993)")

print()
print("=== 2) FP4 expert dequant on real data ===")
for n in ["layers.0.ffn.experts.0.w1", "layers.0.ffn.experts.0.w2", "layers.0.ffn.experts.0.w3"]:
    wq, s = get(n + ".weight"), get(n + ".scale")
    d = m.dequant_fp4_expert(wq.to(dev), s.to(dev), torch.bfloat16)
    print(f"  {n}: packed{tuple(wq.shape)} -> dense{tuple(d.shape)} "
          f"mean|w|={float(d.abs().mean()):.5f} max|w|={float(d.abs().max()):.4f} std={float(d.std()):.5f}")

print()
print("=== 3) FP8 32x32 block dequant on real data ===")
for n in ["layers.0.attn.wq_a", "layers.0.attn.wq_b", "layers.0.attn.wkv",
          "layers.0.attn.wo_a", "layers.0.attn.wo_b", "layers.0.ffn.shared_experts.w1"]:
    wq, s = get(n + ".weight"), get(n + ".scale")
    d = m.dequant_fp8_block(wq.to(dev), s.to(dev), torch.bfloat16)
    print(f"  {n}: {tuple(wq.shape)} scale{tuple(s.shape)} -> dense{tuple(d.shape)} "
          f"mean|w|={float(d.abs().mean()):.5f} max|w|={float(d.abs().max()):.4f} std={float(d.std()):.5f}")

print()
print("=== 4) int4 requant error vs dequantized source ===")
for n, kind in [("layers.0.ffn.experts.0.w1", "fp4"), ("layers.0.attn.wq_a", "fp8")]:
    wq, s = get(n + ".weight"), get(n + ".scale")
    if kind == "fp4":
        d = m.dequant_fp4_expert(wq.to(dev), s.to(dev), torch.float32)
    else:
        d = m.dequant_fp8_block(wq.to(dev), s.to(dev), torch.float32)
    p = m.pack_int4_g32(d, 32)
    rec = unpack(p["weight_packed"], p["weight_scale"], d.shape[0], d.shape[1])
    rel = (rec - d).pow(2).mean().sqrt() / d.pow(2).mean().sqrt()
    cos = torch.nn.functional.cosine_similarity(rec.flatten(), d.flatten(), dim=0)
    print(f"  {n:40s} relRMSE={float(rel):.5f} cos={float(cos):.5f}")

print()
print("=== 5) validate my packer against compressed-tensors' own compressor ===")
from compressed_tensors.compressors import PackedQuantizationCompressor as C
from compressed_tensors.quantization import (QuantizationArgs, QuantizationScheme,
                                             QuantizationStrategy, QuantizationType)
sch = QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(
    num_bits=4, type=QuantizationType.INT, symmetric=True,
    strategy=QuantizationStrategy.GROUP, group_size=32, dynamic=False))
d = m.dequant_fp8_block(get("layers.0.attn.wq_a.weight").to(dev),
                        get("layers.0.attn.wq_a.scale").to(dev), torch.float32)
mine = m.pack_int4_g32(d, 32)
theirs = C.compress({"weight": d.to(torch.bfloat16).cpu(),
                     "weight_scale": mine["weight_scale"].cpu()}, sch)
print("  mine  :", {k: (str(v.dtype), tuple(v.shape)) for k, v in mine.items()})
print("  theirs:", {k: (str(v.dtype), tuple(v.shape)) for k, v in theirs.items() if v is not None})
same = torch.equal(mine["weight_packed"].cpu(), theirs["weight_packed"])
print(f"  weight_packed identical to CT compressor: {same}")
if not same:
    diff = (mine["weight_packed"].cpu() != theirs["weight_packed"]).sum()
    print(f"    differing int32 elements: {int(diff)} / {mine['weight_packed'].numel()}")
