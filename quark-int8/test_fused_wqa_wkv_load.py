#!/usr/bin/env python3
"""Exercise vLLM's real merged-column int4 loading for `attn.fused_wqa_wkv`.

DSV4's loader maps checkpoint tensors into one fused parameter:
    ("attn.fused_wqa_wkv", "attn.wq_a", 0)
    ("attn.fused_wqa_wkv", "attn.wkv",  1)
Both constituents are quantized in our checkpoint, so this checks that
CompressedTensorsWNA16's parameters absorb them with the right shard offsets and
that the assembled buffer equals plain concatenation.

Semantics confirmed from vLLM source: the merged path derives shard_size in
*packed* units from `partitions[shard_id].data.size(1)` and passes it to
ModelWeightParameter.load_merged_column_weight, which adjusts offsets/sizes for
packing internally.
"""
import json

import torch

import vllm.model_executor.parameter as _vp

_vp.get_tensor_model_parallel_rank = lambda: 0
_vp.get_tensor_model_parallel_world_size = lambda: 1

from vllm.model_executor.parameter import (  # noqa: E402
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from safetensors import safe_open  # noqa: E402

PACK_FACTOR = 8
GROUP = 32
OUT = "/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16"
SRC = "/mnt/stripe-3mix-3t2/models/deepseek-ai/DeepSeek-V4.1-Flash"

owm = json.load(open(OUT + "/model.safetensors.index.json"))["weight_map"]
swm = json.load(open(SRC + "/model.safetensors.index.json"))["weight_map"]


def oload(name):
    with safe_open(f"{OUT}/{owm[name]}", "pt") as f:
        return f.get_tensor(name)


def sload(name):
    with safe_open(f"{SRC}/{swm[name]}", "pt") as f:
        return f.get_tensor(name)


SHARDS = [(0, "layers.0.attn.wq_a"), (1, "layers.0.attn.wkv")]

# logical (unpacked) N of each constituent, from the checkpoint itself
logical_n = {sid: oload(mod + ".weight_shape")[0].item() for sid, mod in SHARDS}
logical_k = oload(SHARDS[0][1] + ".weight_shape")[1].item()
n_fused = sum(logical_n.values())
print(f"constituents (unpacked N): { {m: logical_n[s] for s, m in SHARDS} }, K={logical_k}")
print(f"fused: N={n_fused}, K={logical_k}")

# --- parameters exactly as CompressedTensorsWNA16.create_weights builds them
def nop(*a, **k):
    return True


partitions = {
    sid: PackedvLLMParameter(
        input_dim=1, output_dim=0, packed_factor=PACK_FACTOR, packed_dim=1,
        weight_loader=nop,
        data=torch.empty(logical_n[sid], logical_k // PACK_FACTOR, dtype=torch.int32),
    )
    for sid, _ in SHARDS
}
scale = GroupQuantScaleParameter(
    output_dim=0, input_dim=1, weight_loader=nop,
    data=torch.empty(n_fused, logical_k // GROUP, dtype=torch.bfloat16),
)
weight_loader = PackedvLLMParameter(
    input_dim=1, output_dim=0, packed_factor=PACK_FACTOR, packed_dim=1,
    weight_loader=nop,
    data=torch.empty(n_fused, logical_k // PACK_FACTOR, dtype=torch.int32),
)
print("partitions:", {s: tuple(p.shape) for s, p in partitions.items()},
      "| scale", tuple(scale.shape), "| fused", tuple(weight_loader.shape))

# --- load through vLLM's real merged-column path (what _ColumnvLLMParameter
#     does for a PackedvLLMParameter partition)
def load_merged(param, loaded, sid):
    """Call vLLM's real merged-column loader with unpacked offsets/sizes.

    `ModelWeightParameter.load_merged_column_weight` adjusts shard_offset/size
    for packing internally (`_adjust_shard_indexes_for_packing` divides by
    packed_factor), so callers must pass *unpacked* units -- exactly what
    DSV4's `stacked_params_mapping` + LinearBase plumbing does.
    """
    shard_size = logical_n[sid]                       # unpacked
    shard_offset = sum(logical_n[s] for s, _ in SHARDS if s < sid)
    param.load_merged_column_weight(loaded, shard_offset=shard_offset, shard_size=shard_size)
    return shard_size, shard_offset


for sid, mod in SHARDS:
    load_merged(weight_loader, oload(mod + ".weight_packed"), sid)
    load_merged(scale, oload(mod + ".weight_scale"), sid)
    print(f"  loaded shard_id={sid} ({mod})")

exp_pk = torch.cat([oload(m + ".weight_packed") for _, m in SHARDS], 0)
exp_sc = torch.cat([oload(m + ".weight_scale") for _, m in SHARDS], 0).to(torch.bfloat16)
ok_pk = torch.equal(weight_loader, exp_pk)
ok_sc = torch.equal(scale, exp_sc)
print(f"\nfused weight_packed == concat(wq_a, wkv): {ok_pk}")
print(f"fused weight_scale   == concat(...) as bf16: {ok_sc}")
if not ok_pk:
    print("  differing int32:", int((weight_loader != exp_pk).sum()), "/", weight_loader.numel())
if not ok_sc:
    print("  differing scale:", int((scale != exp_sc).sum()), "/", scale.numel())


# --- end-to-end: unpack the fused buffer back to weights and compare with source
def e8m0(s):
    return torch.exp2((s.view(torch.uint8).to(torch.int32) - 127).float())


def ref_fp8(mod):
    w, s = sload(mod + ".weight"), sload(mod + ".scale")
    br, bc = w.shape[0] // s.shape[0], w.shape[1] // s.shape[1]
    return w.float() * e8m0(s).repeat_interleave(br, 0).repeat_interleave(bc, 1)


def unpack(pk, sc):
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    q = ((pk.unsqueeze(-1) >> sh) & 0xF).reshape(pk.shape[0], -1).float() - 8
    return q * sc.repeat_interleave(GROUP, 1)


ref = torch.cat([ref_fp8(m) for _, m in SHARDS], 0)
rec = unpack(weight_loader, scale.float())
rel = ((rec - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
cos = torch.nn.functional.cosine_similarity(rec.flatten(), ref.flatten(), dim=0).item()
print(f"\nfused round-trip vs source: relRMSE={rel:.4f} cos={cos:.4f}")
print("VERDICT:", "PASS" if (ok_pk and ok_sc and rel < 0.15) else "FAIL")
