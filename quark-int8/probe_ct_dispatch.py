#!/usr/bin/env python3
"""Cheap decisive check: does vLLM's compressed-tensors dispatch select the
W4A16 linear scheme (and hence a ROCm W4A16 kernel) for the exact quant config
this converter writes?"""
import json
import sys

import torch
import torch.nn as nn

CFG = sys.argv[1] if len(sys.argv) > 1 else "/w/quark-int8/pilot_ct_int4/config.json"
qc = json.load(open(CFG))["quantization_config"]
print("quant config written by converter:")
print(json.dumps(qc, indent=1)[:900])

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

cfg = CompressedTensorsConfig.from_config(qc)
print("\nparsed:", type(cfg).__name__, "format =", cfg.quant_format)

# Build the QuantizationArgs the same way the config does, then ask the dispatch
# helper which linear scheme it picks.
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy, QuantizationType

wq = QuantizationArgs(num_bits=4, type=QuantizationType.INT, symmetric=True,
                      strategy=QuantizationStrategy.GROUP, group_size=32, dynamic=False)
iq = None
print("\n--- linear scheme dispatch ---")
for fmt in (cfg.quant_format, "pack-quantized"):
    try:
        scheme = cfg._get_scheme(weight_quant=wq, input_quant=iq, format=fmt,
                                 layer_name="layers.0.attn.wq_a")
        print(f"  format={fmt!r} -> {type(scheme).__name__}")
    except Exception as e:
        print(f"  format={fmt!r} -> {type(e).__name__}: {str(e)[:200]}")

print("\n--- MoE scheme dispatch ---")
try:
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        find_matched_target,
    )
    from vllm.model_executor.layers.fused_moe import FusedMoEConfig
    print("  (moe backend selection is performed at layer construction)")
except Exception as e:
    print("  import issue:", e)

print("\n--- ROCm W4A16 kernel availability ---")
from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
from vllm.platforms import PlatformEnum
ks = _POSSIBLE_KERNELS.get(PlatformEnum.ROCM, [])
names = [getattr(k, "__name__", str(k)) for k in ks]
print("  ROCm candidates:", names)
print("  TritonW4A16LinearKernel present:", any("W4A16" in n and "Triton" in n for n in names))
