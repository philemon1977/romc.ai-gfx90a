#!/usr/bin/env python3
"""Quantize Ornith-1.5-397B (Qwen3_5Moe) into a native AMD Quark INT8 checkpoint.

Recipe (no calibration needed, works with Quark file-to-file mode):
  * weight : int8, static, symmetric, per-output-channel (axis 0)
  * input  : int8, DYNAMIC, symmetric, per-token  (qscheme=per_channel + is_dynamic)
  => vLLM QuarkConfig maps this to kInt8StaticChannelSym + kInt8DynamicTokenSym
     - experts -> QuarkW8A8Int8MoEMethod (Triton fused MoE int8)
     - linears -> QuarkW8A8Int8 (AiterInt8ScaledMM / Triton fallback)

Coverage follows Quark's built-in `qwen3_5_moe` LLMTemplate: routed MoE experts are
quantized; attention/GDN/shared-expert/router/MTP/vision/lm_head stay BF16.
Fused expert tensors (mlp.experts.gate_up_proj / down_proj) are unfused into
per-expert tensors by the template's f2f_weight_converters (SplitFusedExperts).

Memory: processes one safetensors shard at a time; peak device usage = 1 shard.
"""

import argparse
import time

import torch  # noqa: F401  (kept: quark observers reference torch dtypes)

from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import (
    Int8PerChannelSpec,
    QConfig,
    QLayerConfig,
    QTensorConfig,
)
from quark.torch.quantization.config.template import LLMTemplate
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import PerChannelMinMaxObserver


def build_qconfig(model_type: str, coverage: str = "experts", weight_dtype: str = "int8"):
    """coverage='experts' -> AMD LLMTemplate defaults (routed experts only).
    coverage='experts_attn' -> additionally quantize the attention / GDN projections
    so the INT8 GEMMs actually reach aiter's gemm_a8w8 (gfx90a JIT) path.
    weight_dtype='int8'   -> W8A8: int8 per-channel weights + dynamic per-token int8 acts
    weight_dtype='mxfp4'  -> W4A16 weight-only MXFP4 (per-group 32, e8m0 scales, bf16 acts)
                             -> vLLM QuarkOCP_MX / QuarkOCP_MX_MoEMethod
    """
    tmpl = LLMTemplate.get(model_type)

    if weight_dtype == "int8":
        weight = Int8PerChannelSpec(
            ch_axis=0,
            symmetric=True,
            scale_type="float",
            round_method="half_even",
            is_dynamic=False,
        ).to_quantization_spec()
        # dynamic per-token activation spec: per_channel + is_dynamic is exactly what
        # vLLM's _is_w8a8_int8() reads as "int8 dynamic token" (no stored input scale).
        input_dyn = QTensorConfig(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_channel,
            ch_axis=-1,
            symmetric=True,
            is_dynamic=True,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            observer_cls=PerChannelMinMaxObserver,
        )
    elif weight_dtype == "mxfp4":
        from quark.torch.quantization.config.config import OCP_MXFP4Spec

        weight = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
        input_dyn = None  # weight-only: activations stay bf16
    else:
        raise SystemExit(f"unknown weight dtype: {weight_dtype}")

    if coverage == "experts":
        exclude = list(tmpl.exclude_layers_name)
    elif coverage == "experts_attn":
        # start from the template, then re-admit the projection linears while keeping
        # everything numerically sensitive / non-Linear excluded.
        exclude = [
            "lm_head",
            "model.visual.*",
            "mtp.*",
            "*mlp.gate", "*mlp.gate.linear",
            "*shared_expert_gate*",
            "*.shared_expert.*",
            "*.linear_attn.conv1d", "*.linear_attn.conv2d", "*.linear_attn.convNd",
            "*.linear_attn.in_proj_a", "*.linear_attn.in_proj_b",
            "*.linear_attn.A_log", "*.linear_attn.dt_bias", "*.linear_attn.norm",
        ]
    else:
        raise SystemExit(f"unknown coverage: {coverage}")

    qcfg = QConfig(
        global_quant_config=QLayerConfig(weight=weight, input_tensors=input_dyn),
        exclude=exclude,
    )
    converters = list(getattr(tmpl, "f2f_weight_converters", None) or [])
    return qcfg, converters, tmpl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="source BF16 checkpoint dir")
    ap.add_argument("--out", required=True, help="output Quark INT8 checkpoint dir")
    ap.add_argument("--device", default="cuda", help="cuda | cuda:N | cpu")
    ap.add_argument("--model-type", default="qwen3_5_moe")
    ap.add_argument("--coverage", default="experts", choices=["experts", "experts_attn"],
                    help="experts = AMD template (routed experts only); "
                         "experts_attn = also quantize self_attn/linear_attn projections")
    ap.add_argument("--weight-dtype", default="int8", choices=["int8", "mxfp4"],
                    help="int8 = W8A8 int8 per-channel + dynamic per-token acts; "
                         "mxfp4 = W4A16 weight-only MXFP4 (per-group 32, e8m0 scales)")
    ap.add_argument("--allow-default-exclude", action="store_true",
                    help="use even if template has no excludes (sanity opt-in)")
    args = ap.parse_args()

    qcfg, converters, tmpl = build_qconfig(args.model_type, args.coverage, args.weight_dtype)
    if not tmpl.exclude_layers_name and not args.allow_default_exclude:
        raise SystemExit("template has empty exclude list; refusing to run")
    if not converters:
        raise SystemExit("template provides no f2f_weight_converters; fused experts would not be quantized")

    print(f"[quantize] model      : {args.model}")
    print(f"[quantize] out        : {args.out}")
    print(f"[quantize] device     : {args.device}")
    print(f"[quantize] coverage   : {args.coverage}")
    print(f"[quantize] weight     : {args.weight_dtype}")
    print(f"[quantize] exclude    : {qcfg.exclude}")
    print(f"[quantize] converters : {[(c.source_patterns, c.target_patterns) for c in converters]}")

    t0 = time.time()
    quantizer = ModelQuantizer(qcfg)
    quantizer.direct_quantize_checkpoint(
        pretrained_model_path=args.model,
        save_path=args.out,
        weight_converters=converters,
        device=args.device,
    )
    print(f"[quantize] DONE in {(time.time() - t0) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
