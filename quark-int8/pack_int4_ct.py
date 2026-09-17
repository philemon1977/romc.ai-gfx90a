#!/usr/bin/env python3
"""Stream a BF16 checkpoint into an INT4 W4A16 checkpoint in **compressed-tensors
`pack-quantized`** format, so vLLM on gfx90a can use the TRITON W4A16 kernels
(`_POSSIBLE_KERNELS[ROCM]` -> TritonW4A16LinearKernel, MoE oracle int_wna16 ->
WNA16MoEBackend.TRITON) instead of the MXFP4 EMULATION backend that made v3 slow.

Why not Quark: vLLM's QuarkConfig has no int4 weight-only scheme
(`NotImplementedError: No quark compatible scheme was found` for int4/uint4
per_group|per_channel), and the W4A16 kernels are consumed by
auto_awq / auto_gptq / compressed-tensors / moe_wna16 only.

Layout written per quantized 2D weight (compressed-tensors conventions):
    <name>.weight_packed  int32 [N, K/(32/num_bits)]   (8 x int4 per int32)
    <name>.weight_scale   float32 [N, K/group_size]
    <name>.weight_shape   int64 [2] = (N, K)
Quantization: symmetric int4 round-to-nearest, per-group along K (input dim).

Same expert handling as the Quark runs: fused `mlp.experts.gate_up_proj`
[E, 2I, H] / `down_proj` [E, H, I] are split into per-expert tensors
(gate = first half of rows, up = second half), everything excluded from the
recipe stays BF16.
"""
import argparse
import fnmatch
import json
import os
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compressed_tensors.compressors import PackedQuantizationCompressor
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)

EXCLUDE_EXPERTS = [
    "lm_head", "model.visual.*", "mtp.*",
    "*mlp.gate", "*mlp.gate.linear", "*shared_expert_gate*", "*.shared_expert.*",
    "*.linear_attn.*", "*.self_attn.*",
]
EXCLUDE_EXPERTS_ATTN = [
    "lm_head", "model.visual.*", "mtp.*",
    "*mlp.gate", "*mlp.gate.linear", "*shared_expert_gate*", "*.shared_expert.*",
    "*.linear_attn.conv1d", "*.linear_attn.conv2d", "*.linear_attn.in_proj_a",
    "*.linear_attn.in_proj_b", "*.linear_attn.A_log", "*.linear_attn.dt_bias",
    "*.linear_attn.norm",
]


def scheme_for(group_size: int) -> QuantizationScheme:
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4, type=QuantizationType.INT, symmetric=True,
            strategy=QuantizationStrategy.GROUP, group_size=group_size, dynamic=False,
        ),
    )


def should_quantize(name: str, tensor: torch.Tensor, core: str, exclude: list[str]) -> bool:
    if not name.endswith(".weight") or tensor.ndim != 2:
        return False
    if core.endswith("norm") or "embed" in core or ".conv" in core:
        return False
    return not any(fnmatch.fnmatch(core, p) for p in exclude)


def quantize_and_pack(w: torch.Tensor, group_size: int, comp: type, scheme) -> dict:
    N, K = w.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    wg = w.float().reshape(N, K // group_size, group_size)
    scale = (wg.abs().amax(-1, keepdim=True) / 7.0).clamp_min(1e-8)  # symmetric int4, qmax=7
    return comp.compress({"weight": w, "weight_scale": scale.squeeze(-1).contiguous()}, scheme)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--coverage", default="experts", choices=["experts", "experts_attn"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    exclude = EXCLUDE_EXPERTS if args.coverage == "experts" else EXCLUDE_EXPERTS_ATTN
    scheme = scheme_for(args.group_size)
    comp = PackedQuantizationCompressor
    dev = torch.device(args.device)

    os.makedirs(args.out, exist_ok=True)
    shards = sorted(f for f in os.listdir(args.model) if f.endswith(".safetensors"))
    print(f"[pack] {len(shards)} shards, group_size={args.group_size}, coverage={args.coverage}, device={args.device}")

    index: dict[str, str] = {}
    split_experts = 0
    quantized_tensors = 0
    t0 = time.time()
    for si, shard in enumerate(shards, 1):
        with safe_open(os.path.join(args.model, shard), framework="pt") as f:
            items = {k: f.get_tensor(k) for k in f.keys()}
        out: dict[str, torch.Tensor] = {}
        for name, tensor in items.items():
            core = name.rsplit(".", 1)[0]
            # fused MoE experts (NO .weight suffix) -> per-expert tensors
            if name.endswith(".mlp.experts.gate_up_proj") or name.endswith(".mlp.experts.down_proj"):
                prefix = name.rsplit(".", 1)[0]
                is_gate_up = name.endswith("gate_up_proj")
                t = tensor.to(dev)
                n_exp = t.shape[0]
                if is_gate_up:
                    half = t.shape[1] // 2
                    for e in range(n_exp):
                        out[f"{prefix}.{e}.gate_proj.weight"] = t[e, :half]
                        out[f"{prefix}.{e}.up_proj.weight"] = t[e, half:]
                else:
                    for e in range(n_exp):
                        out[f"{prefix}.{e}.down_proj.weight"] = t[e]
                split_experts += n_exp
                continue
            out[name] = tensor

        quantized: dict[str, torch.Tensor] = {}
        for name, tensor in out.items():
            core = name.rsplit(".", 1)[0]
            if should_quantize(name, tensor, core, exclude):
                packed = quantize_and_pack(tensor.to(dev), args.group_size, comp, scheme)
                base = name[: -len(".weight")]
                for k, v in packed.items():  # weight_packed / weight_scale / weight_shape
                    quantized[f"{base}.{k}"] = v.to("cpu") if k != "weight_shape" else v
                quantized_tensors += 1
            else:
                quantized[name] = tensor
        save_file(quantized, os.path.join(args.out, shard), metadata={"format": "pt"})
        for k in quantized:
            index[k] = shard
        print(f"[pack] {si}/{len(shards)} {shard}: {len(quantized)} tensors "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)

    print(f"[pack] split {split_experts} fused expert tensors; quantized {quantized_tensors} 2D weights")
    if args.coverage.startswith("experts") and quantized_tensors == 0:
        raise SystemExit("[pack] ERROR: nothing was quantized -- fused-expert matching is broken")

    json.dump({"metadata": {"total_size": 0}, "weight_map": index},
              open(os.path.join(args.out, "model.safetensors.index.json"), "w"), indent=1)

    # assets + config
    for fn in os.listdir(args.model):
        src = os.path.join(args.model, fn)
        if fn.endswith(".safetensors") or fn.startswith("model.safetensors.index"):
            continue
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(args.out, fn), dirs_exist_ok=True)
        else:
            shutil.copy2(src, os.path.join(args.out, fn))
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "input_activations": None,
                "weights": {
                    "num_bits": 4, "type": "int", "symmetric": True,
                    "strategy": "group", "group_size": args.group_size,
                    "dynamic": False, "actorder": None,
                },
            }
        },
        "ignore": exclude + ["re:.*embed_tokens.*", "re:.*norm.*", "re:.*conv.*"],
        "quantization_status": "compressed",
    }
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    print(f"[pack] DONE in {(time.time()-t0)/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
