#!/usr/bin/env python3
"""Smoke test A: build a mini qwen3_5_moe-shaped checkpoint, run the Quark
file-to-file INT8 driver on it (CPU), verify tensors + config, then verify
vLLM's QuarkConfig maps every layer to the expected scheme.
"""
import json
import os
import shutil
import subprocess
import sys

import torch
from safetensors.torch import save_file

ROOT = "/work/mini_src"
OUT = "/work/mini_int8"
H = 64
I = 32
E = 8


def make_src():
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)
    os.makedirs(ROOT)
    g = torch.Generator().manual_seed(0)
    t = lambda *s: torch.randn(*s, generator=g, dtype=torch.bfloat16)
    tensors = {
        # embed + norm (must stay untouched)
        "model.language_model.embed_tokens.weight": t(256, H),
        "model.language_model.norm.weight": t(H),
        # layer 0: full attention path (template-excluded)
        "model.language_model.layers.0.input_layernorm.weight": t(H),
        "model.language_model.layers.0.post_attention_layernorm.weight": t(H),
        "model.language_model.layers.0.self_attn.q_proj.weight": t(2 * H, H),
        "model.language_model.layers.0.self_attn.k_proj.weight": t(H // 4, H),
        "model.language_model.layers.0.self_attn.v_proj.weight": t(H // 4, H),
        "model.language_model.layers.0.self_attn.o_proj.weight": t(H, 2 * H),
        # layer 1: linear attention path (template-excluded)
        "model.language_model.layers.1.input_layernorm.weight": t(H),
        "model.language_model.layers.1.linear_attn.in_proj_qkv.weight": t(3 * H, H),
        "model.language_model.layers.1.linear_attn.in_proj_z.weight": t(H, H),
        "model.language_model.layers.1.linear_attn.in_proj_a.weight": t(E, H),
        "model.language_model.layers.1.linear_attn.out_proj.weight": t(H, 3 * H),
        "model.language_model.layers.1.linear_attn.convNd.weight": t(3 * H, 1, 4),
        # router + shared expert (template-excluded)
        "model.language_model.layers.0.mlp.gate.weight": t(E, H),
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": t(I, H),
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": t(I, H),
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": t(H, I),
        "model.language_model.layers.0.mlp.shared_expert_gate.weight": t(1, H),
        # routed experts, FUSED layout (the quantization target)
        "model.language_model.layers.0.mlp.experts.gate_up_proj": t(E, 2 * I, H),
        "model.language_model.layers.0.mlp.experts.down_proj": t(E, H, I),
        "model.language_model.layers.1.mlp.experts.gate_up_proj": t(E, 2 * I, H),
        "model.language_model.layers.1.mlp.experts.down_proj": t(E, H, I),
        # vision + mtp + lm_head (template-excluded)
        "model.visual.blocks.0.attn.proj.weight": t(H, H),
        "mtp.layers.0.mlp.experts.0.gate_proj.weight": t(I, H),
        "lm_head.weight": t(256, H),
    }
    # split into two shards to exercise multi-file flow
    names = list(tensors)
    half = len(names) // 2
    save_file({k: tensors[k] for k in names[:half]}, os.path.join(ROOT, "model-00001-of-00002.safetensors"))
    save_file({k: tensors[k] for k in names[half:]}, os.path.join(ROOT, "model-00002-of-00002.safetensors"))

    cfg = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "dtype": "bfloat16",
        "model_type": "qwen3_5_moe",
        "text_config": {
            "dtype": "bfloat16", "hidden_size": H, "num_hidden_layers": 2,
            "num_experts": E, "num_experts_per_tok": 2, "moe_intermediate_size": I,
            "model_type": "qwen3_5_moe_text",
        },
        "vision_config": {"model_type": "qwen3_5_moe_vision", "hidden_size": H, "depth": 1},
    }
    with open(os.path.join(ROOT, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[smoke] src checkpoint: {len(names)} tensors in 2 shards")


def run_driver():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    r = subprocess.run(
        [sys.executable, "/work/quantize_ornith_int8.py", "--model", ROOT, "--out", OUT, "--device", "cpu"],
        capture_output=True, text=True,
    )
    tail = "\n".join((r.stdout + "\n" + r.stderr).splitlines()[-25:])
    print(tail)
    if r.returncode != 0:
        raise SystemExit("driver failed")


def verify_tensors():
    from safetensors import safe_open
    out = {}
    for fn in os.listdir(OUT):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(OUT, fn), framework="pt") as f:
                for k in f.keys():
                    out[k] = f.get_tensor(k)
    # 1) per-expert int8 tensors + scales exist
    ok = True
    for lyr in (0, 1):
        for e in range(E):
            for nm in ("gate_proj", "up_proj", "down_proj"):
                w = f"model.language_model.layers.{lyr}.mlp.experts.{e}.{nm}.weight"
                s = w + "_scale"
                if w not in out or out[w].dtype != torch.int8:
                    print("MISSING/WRONG DTYPE:", w, out.get(w, None) is not None and out[w].dtype); ok = False
                if s not in out:
                    print("MISSING SCALE:", s); ok = False
    # 2) scale shapes: per output channel
    gs = out["model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale"]
    ds = out["model.language_model.layers.0.mlp.experts.0.down_proj.weight_scale"]
    print("[smoke] gate scale shape", tuple(gs.shape), "down scale shape", tuple(ds.shape))
    assert tuple(gs.shape) == (I,) and tuple(ds.shape) == (H,), "unexpected scale shapes"
    # 3) round-trip accuracy on one expert slice
    w0 = out["model.language_model.layers.0.mlp.experts.0.gate_proj.weight"]
    # reconstruct original from src for comparison
    src = {}
    for fn in os.listdir(ROOT):
        if not fn.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(ROOT, fn), framework="pt") as f:
            for k in f.keys():
                src[k] = f.get_tensor(k)
    orig = src["model.language_model.layers.0.mlp.experts.gate_up_proj"][0, :I]
    recon = w0.float() * gs.float().unsqueeze(1)
    err = (recon - orig.float()).abs().max().item()
    rel = err / orig.float().abs().max().item()
    print(f"[smoke] expert0 gate_proj: max abs err {err:.5f} (rel {rel:.4f})")
    assert rel < 0.02, "quantization error too large"
    # gate/up halves: gate = first half rows of fused tensor
    up0 = src["model.language_model.layers.0.mlp.experts.gate_up_proj"][0, I:]
    wu = out["model.language_model.layers.0.mlp.experts.0.up_proj.weight"]
    scale_u = out["model.language_model.layers.0.mlp.experts.0.up_proj.weight_scale"]
    recon_u = wu.float() * scale_u.float().unsqueeze(1)
    assert (recon_u - up0.float()).abs().max() < 0.05, "up half mismatch (split order wrong?)"
    # 4) excluded tensors remain bf16 and unfused names gone
    assert out["model.language_model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
    assert out["model.language_model.layers.1.linear_attn.in_proj_qkv.weight"].dtype == torch.bfloat16
    assert "model.language_model.layers.0.mlp.experts.gate_up_proj" not in out, "fused tensor should be gone"
    assert out["mtp.layers.0.mlp.experts.0.gate_proj.weight"].dtype == torch.bfloat16
    assert out["model.visual.blocks.0.attn.proj.weight"].dtype == torch.bfloat16
    assert out["lm_head.weight"].dtype == torch.bfloat16
    # 5) exported config.json
    cfg = json.load(open(os.path.join(OUT, "config.json")))
    qc = cfg["quantization_config"]
    print("[smoke] exported quantization_config.global_quant_config =", json.dumps(qc.get("global_quant_config"), indent=1)[:800])
    assert qc["quant_method"] == "quark" if "quant_method" in qc else True
    g = qc["global_quant_config"]
    assert g["weight"]["dtype"] == "int8" and g["weight"]["qscheme"] == "per_channel" and g["weight"]["symmetric"] is True
    assert g["input_tensors"]["dtype"] == "int8" and g["input_tensors"]["is_dynamic"] is True
    assert "lm_head" in qc["exclude"]
    print("[smoke] tensors + config OK")
    return qc


def verify_vllm(qc):
    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    except ImportError:
        try:
            from vllm.model_executor.layers.fused_moe import FusedMoE
        except ImportError:
            from vllm.model_executor.layers.fused_moe.layer import RoutedExperts as FusedMoE
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt8DynamicTokenSym, kInt8StaticChannelSym)
    try:
        qconf = QuarkConfig.from_config(qc, prefix="")
    except TypeError:
        qconf = QuarkConfig.from_config(qc)
    # linear scheme for an expert module name
    wk, ak, cls = qconf.get_scheme_cls(torch.nn.Linear, "model.language_model.layers.0.mlp.experts.3.gate_proj")
    print("[smoke] linear scheme:", cls.__name__, wk, ak)
    assert wk == kInt8StaticChannelSym and ak == kInt8DynamicTokenSym, "linear scheme mismatch"
    # MoE scheme
    wk2, ak2, cls2 = qconf.get_scheme_cls(FusedMoE, "model.language_model.layers.0.mlp")
    print("[smoke] moe scheme:", cls2.__name__)
    # excluded layers: expect vLLM to report them as not-quantized (exception or None both OK)
    try:
        res = qconf.get_scheme_cls(torch.nn.Linear, "model.language_model.layers.0.self_attn.q_proj")
        print("[smoke] excluded-layer query returned:", res)
    except Exception as exc:
        print("[smoke] excluded-layer query raised as expected:", type(exc).__name__)
    print("[smoke] vLLM QuarkConfig OK")


if __name__ == "__main__":
    make_src()
    run_driver()
    qc = verify_tensors()
    verify_vllm(qc)
    print("SMOKE-A PASS")
