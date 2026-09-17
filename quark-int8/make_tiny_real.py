#!/usr/bin/env python3
"""Build a TINY but architecturally faithful Qwen3_5Moe checkpoint using the
installed (patched) transformers, then it can be quantized with the same
Quark driver and served by vLLM to validate the full INT8 integration loop
in seconds instead of the 397B's ~40 min weight load."""
import argparse
import json
import os

import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B",
                    help="reference checkpoint dir (for tokenizer files)")
    ap.add_argument("--out", default="/work/tiny_src")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from transformers import Qwen3_5MoeConfig  # patched transformers 5.x
    from transformers import Qwen3_5MoeForConditionalGeneration

    ref_cfg = json.load(open(os.path.join(args.ref, "config.json")))
    rt = ref_cfg["text_config"]
    rv = ref_cfg["vision_config"]

    cfg = Qwen3_5MoeConfig(
        dtype="bfloat16",
        text_config=dict(
            **{k: v for k, v in rt.items() if k not in (
                "layer_types", "num_hidden_layers", "hidden_size", "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size", "shared_expert_intermediate_size",
                "num_attention_heads", "num_key_value_heads", "head_dim",
                "linear_num_key_heads", "linear_key_head_dim",
                "linear_num_value_heads", "linear_value_head_dim",
                "linear_conv_kernel_dim", "max_position_embeddings", "vocab_size",
                "mtp_num_hidden_layers", "rope_parameters")},
            num_hidden_layers=args.layers,
            layer_types=(["linear_attention", "linear_attention", "linear_attention", "full_attention"]
                         * ((args.layers + 3) // 4))[:args.layers],
            hidden_size=256,
            num_attention_heads=8, num_key_value_heads=2, head_dim=32,
            linear_num_key_heads=4, linear_key_head_dim=32,
            linear_num_value_heads=8, linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            num_experts=16, num_experts_per_tok=4,
            moe_intermediate_size=128, shared_expert_intermediate_size=128,
            max_position_embeddings=4096, vocab_size=rt["vocab_size"],  # keep real tokenizer vocab
            # head_dim=32 * partial_rotary_factor 0.25 -> rotary_dim 8 -> mrope_section must sum to 4
            rope_parameters=dict(rt["rope_parameters"], mrope_section=[1, 1, 2]),
            mtp_num_hidden_layers=1,
        ),
        vision_config=dict(
            **{k: v for k, v in rv.items() if k not in ("depth", "hidden_size", "intermediate_size", "out_hidden_size", "num_heads")},
            depth=2, hidden_size=128, intermediate_size=256, num_heads=4, out_hidden_size=256,
        ),
    )
    torch.manual_seed(args.seed)
    model = Qwen3_5MoeForConditionalGeneration(cfg)
    model = model.to(torch.bfloat16)
    model.save_pretrained(args.out, safe_serialization=True)

    # copy tokenizer assets from the real checkpoint so vLLM can load them
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
               "chat_template.jinja", "generation_config.json",
               "preprocessor_config.json", "video_preprocessor_config.json",
               "processor_config.json"):
        src = os.path.join(args.ref, fn)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, os.path.join(args.out, fn))
    n = sum(1 for f in os.listdir(args.out) if f.endswith(".safetensors"))
    sz = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out) if f.endswith(".safetensors"))
    print(f"[tiny] saved {n} shards, {sz/1e6:.1f} MB -> {args.out}")

if __name__ == "__main__":
    main()
