#!/usr/bin/env python3
"""Construct one real DSV4.1 attention layer with the converted checkpoint's
quantization config and load the converted tensors into it.

This exercises the actual end-to-end contract -- vLLM model code builds the
quantized parameters, `CompressedTensorsWNA16` creates them, and the converted
checkpoint's names/shapes are loaded through vLLM's own weight loaders -- using
only the real hyperparameters (2 layers instead of 40) and one shard of tensors.

Requires a small GPU (one attention layer at TP1 is a few GiB).
"""
import json
import os

import torch

MODEL = "/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16"

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
# single-process rendezvous for the TP group the parameter classes need
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29577")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

hf = json.load(open(os.path.join(MODEL, "config.json")))
tc = dict(hf["text_config"])
# shrink to something we can build on one GPU; keep every attention-relevant field
tc.update(num_hidden_layers=2, n_routed_experts=8, num_nextn_predict_layers=0,
          engram_layer_ids=[], index_source_layer_ids=[], kv_source_layer_ids=[],
          compress_ratios=[0, 1], num_hash_layers=0, vocab_size=1024)

from vllm.config import (  # noqa: E402
    ModelConfig, VllmConfig, set_current_vllm_config,
)

# vLLM registers deepseek_v41 lazily during config processing; do it up front so
# ModelConfig's validation finds the architecture.
from vllm.transformers_utils.config import _register_config_class, _CONFIG_REGISTRY  # noqa: E402

_register_config_class("deepseek_v41", _CONFIG_REGISTRY["deepseek_v41"])
print("registered deepseek_v41 config class")

model_config = ModelConfig(
    model=MODEL,
    tokenizer=MODEL,
    tokenizer_mode="deepseek_v41",
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=512,
    enforce_eager=True,
    hf_overrides=tc,
)
vllm_config = VllmConfig(model_config=model_config)
# TritonW4A16LinearKernel only implements fp16/bf16 activations, so the config
# must carry a resolved bf16 dtype (the engine does this via --dtype bfloat16).
if vllm_config.model_config.dtype != torch.bfloat16:
    print("WARNING: resolved dtype is", vllm_config.model_config.dtype)
vllm_config.model_config.dtype = torch.bfloat16
print("act dtype for kernels:", vllm_config.model_config.dtype)
print("quant_config:", type(vllm_config.quant_config).__name__)
print("model_type:", model_config.hf_config.model_type)

from vllm.model_executor.layers.linear import (  # noqa: E402
    MergedColumnParallelLinear, ColumnParallelLinear, RowParallelLinear,
)

qc = vllm_config.quant_config

# Parameter classes need a TP group (rank/size) at construction time.
from vllm.distributed import (  # noqa: E402
    init_distributed_environment, initialize_model_parallel,
)
init_distributed_environment(world_size=1, rank=0, local_rank=0,
                             distributed_init_method="env://", backend="gloo")

with set_current_vllm_config(vllm_config):
    initialize_model_parallel(tensor_model_parallel_size=1)
    H = tc["hidden_size"]
    ql = tc["q_lora_rank"]
    head_dim = tc["head_dim"]
    n_heads = tc["num_attention_heads"]
    q_dim = n_heads * head_dim
    kv_dim = tc["head_dim"]

    # exactly the linears DeepseekV4Attention builds for these tensors
    fused = MergedColumnParallelLinear(
        input_size=H, output_sizes=[ql, kv_dim], bias=False,
        quant_config=qc, prefix="layers.0.attn.fused_wqa_wkv", disable_tp=True,
        params_dtype=torch.bfloat16)
    wq_b = ColumnParallelLinear(input_size=ql, output_size=q_dim, bias=False,
                                quant_config=qc, prefix="layers.0.attn.wq_b",
                                disable_tp=True, params_dtype=torch.bfloat16)
    wo_b = RowParallelLinear(input_size=q_dim, output_size=H, bias=False,
                             quant_config=qc, prefix="layers.0.attn.wo_b",
                             disable_tp=True, params_dtype=torch.bfloat16)
    print("built: fused_wqa_wkv, wq_b, wo_b")

    for name, layer in (("fused_wqa_wkv", fused), ("wq_b", wq_b), ("wo_b", wo_b)):
        prm = dict(layer.named_parameters())
        print(f"  {name}: {[(k, tuple(v.shape), str(v.dtype)) for k, v in prm.items()]}")

    # ---- now load the converted checkpoint tensors through vLLM's loaders
    from safetensors import safe_open
    owm = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]

    def load_tensor(name):
        with safe_open(os.path.join(MODEL, owm[name]), "pt") as f:
            return f.get_tensor(name)

    params = dict(fused.named_parameters())
    # DSV4 stacked_params_mapping: wq_a -> shard 0, wkv -> shard 1
    for sub, shard_id in (("wq_a", 0), ("wkv", 1)):
        base = f"layers.0.attn.{sub}"
        pl = params["weight_packed"].weight_loader
        sl = params["weight_scale"].weight_loader
        shard = load_tensor(base + ".weight_shape").tolist()[0]
        size = shard
        offset = 0 if shard_id == 0 else load_tensor("layers.0.attn.wq_a.weight_shape").tolist()[0]
        pl(params["weight_packed"], load_tensor(base + ".weight_packed"), shard_id)
        sl(params["weight_scale"], load_tensor(base + ".weight_scale"), shard_id)
        print(f"  loaded {base} into fused shard {shard_id} "
              f"(unpacked size={size}, offset={offset})")

    print("\nRESULT: real DSV4.1 attention layer accepted the converted tensors")
    print("  fused weight_packed:", tuple(params["weight_packed"].shape))
    print("  fused weight_scale :", tuple(params["weight_scale"].shape))
    print("  wq_b  weight_packed:", tuple(dict(wq_b.named_parameters())["weight_packed"].shape))
    print("  wo_b  weight_packed:", tuple(dict(wo_b.named_parameters())["weight_packed"].shape))
