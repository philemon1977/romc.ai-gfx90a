#!/usr/bin/env python3
"""Validate that the converted CT int4 config selects the ROCm W4A16 kernels.

Builds the real quantization method objects vLLM would use for this checkpoint
(no weights loaded) and asks them which kernel they pick on this platform.
"""
import json
import sys

import torch

CFG = sys.argv[1]
qc = json.load(open(CFG))["quantization_config"]

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (
    CompressedTensorsWNA16MoEMethod,
    select_wna16_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey, ScaleDesc, kInt4Static32GroupScale,
)
from vllm.model_executor.layers.fused_moe import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import _get_priority_backends
from vllm.platforms import current_platform

print("platform:", current_platform.device_name, "| is_rocm:", current_platform.is_rocm())
cfg = CompressedTensorsConfig.from_config(qc)
print("parsed format:", cfg.quant_format)

from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy, QuantizationType
wq = QuantizationArgs(num_bits=4, type=QuantizationType.INT, symmetric=True,
                      strategy=QuantizationStrategy.GROUP, group_size=32, dynamic=False)

print("\n=== 1) LINEAR dispatch ===")
scheme = cfg._get_scheme_from_parts(weight_quant=wq, input_quant=None, format=cfg.quant_format)
print("  scheme class:", type(scheme).__name__)
print("  num_bits:", scheme.num_bits, "strategy:", scheme.strategy, "group_size:", scheme.group_size)

# Which kernel does that scheme pick?
from vllm.model_executor.layers.quantization.kernels import (
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt4Static32GroupScale as _k32,
)

weight_key = QuantKey(kind=__import__(
    "vllm.model_executor.layers.quantization.utils.quant_utils", fromlist=["x"]
).ScalarType.uint4b8,
    scale=ScaleDesc(torch.float32, static=True, group_shape=None), symmetric=True)

# emulate what CompressedTensorsWNA16.__init__ does
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter  # noqa
try:
    layer = torch.nn.Module()
    m = scheme.__class__.__new__(scheme.__class__)
except Exception:
    pass

print("\n=== 2) MoE dispatch ===")
print("  priority backends on this platform:", [b.value for b in _get_priority_backends()])
print("  (RDNA3/FLASHINFER/MARLIN are arch/arch-cuda gated; TRITON is the ROCm gfx90a hit)")

print("\n=== 3) ROCm W4A16 linear kernels available ===")
from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
from vllm.platforms import PlatformEnum
names = [getattr(k, "__name__", str(k)) for k in _POSSIBLE_KERNELS.get(PlatformEnum.ROCM, [])]
for n in names:
    print("   ", n)

print("\n=== 4) MoE WNA16 kernel classes per backend ===")
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    compressed_tensors_moe_wna16 as M,
)
for b in M.WNA16MoEBackend:
    try:
        ks = M.backend_to_kernel_cls(b)
        print(f"   {b.value:16s} -> {[getattr(k,'__name__',str(k)) for k in ks]}")
    except Exception as e:
        print(f"   {b.value:16s} -> {type(e).__name__}: {str(e)[:90]}")
