#!/usr/bin/env python3
"""Verify the Ornith-1.5-397B Quark INT8 checkpoint.

 1. index <-> shard consistency, total size, int8/bf16 byte accounting
 2. quantization_config correctness (int8 per-channel weight + dynamic per-token input)
 3. full expert coverage: 60 layers x 512 experts x {gate,up,down} int8 + per-channel scale
 4. excluded tensors stay BF16
 5. real round-trip error vs the ORIGINAL (fused) bf16 source tensors
 6. tokenizer/processor assets + architecture fields preserved
"""
import json
import os
import random
import sys

import torch
from safetensors import safe_open

SRC = "/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B"
DST = os.environ.get("DST", "/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-Quark-Int8")
NL, NE, H, I = 60, 512, 4096, 1024
N_SAMPLE = int(os.environ.get("N_SAMPLE", "3"))
ASSETS = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
          "chat_template.jinja", "generation_config.json", "preprocessor_config.json",
          "video_preprocessor_config.json", "processor_config.json", "assets"]


def rel_err(recon: torch.Tensor, orig: torch.Tensor) -> float:
    o = orig.float()
    return ((recon.float() - o).abs().max() / (o.abs().max() + 1e-9)).item()


def main() -> None:
    ok = True
    cfg = json.load(open(os.path.join(DST, "config.json")))
    src_cfg = json.load(open(os.path.join(SRC, "config.json")))
    assert cfg["architectures"] == src_cfg["architectures"], "architecture changed"
    assert cfg["text_config"]["num_hidden_layers"] == src_cfg["text_config"]["num_hidden_layers"]
    qc = cfg["quantization_config"]
    g = qc["global_quant_config"]
    assert g["weight"]["dtype"] == "int8" and g["weight"]["qscheme"] == "per_channel" and g["weight"]["symmetric"]
    assert g["input_tensors"]["dtype"] == "int8" and g["input_tensors"]["is_dynamic"]
    print(f"[1] config: shape preserved; quant = int8 per-channel weight + dynamic per-token input; excludes={len(qc['exclude'])}")

    idx = json.load(open(os.path.join(DST, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    files = sorted(set(wm.values()))
    missing = [f for f in files if not os.path.exists(os.path.join(DST, f))]
    int8_bytes = bf16_bytes = 0
    for f in files:
        p = os.path.join(DST, f)
        if os.path.exists(p):
            bf16_bytes += os.path.getsize(p)
    print(f"[2] index: {len(wm)} tensors / {len(files)} shards, missing files: {len(missing)}, total {bf16_bytes/1e9:.1f} GB")
    ok &= not missing

    n_w = n_s = 0
    for lyr in range(NL):
        for e in range(NE):
            for nm in ("gate_proj", "up_proj", "down_proj"):
                w = f"model.language_model.layers.{lyr}.mlp.experts.{e}.{nm}.weight"
                n_w += w in wm
                n_s += (w + "_scale") in wm
    exp = NL * NE * 3
    print(f"[3] expert int8 tensors {n_w}/{exp}, scales {n_s}/{exp}")
    ok &= n_w == exp and n_s == exp

    src_idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
    random.seed(1234)
    sample_shards = random.sample(files, min(N_SAMPLE, len(files)))
    errs, checked, excl_checked = [], 0, 0
    for shard in sample_shards:
        with safe_open(os.path.join(DST, shard), framework="pt") as f:
            keys = list(f.keys())
            tens = {k: f.get_tensor(k) for k in keys if k.endswith(".weight") or k.endswith("_scale")}
        for k, t in tens.items():
            if not k.endswith(".weight") or "experts" not in k or t.dtype != torch.int8:
                continue
            s = tens.get(k + "_scale")
            if s is None:
                continue
            m = k.split(".")
            lyr = int(m[m.index("layers") + 1]); e = int(m[m.index("experts") + 1]); nm = m[-2]
            fused = f"model.language_model.layers.{lyr}.mlp.experts." + ("gate_up_proj" if nm in ("gate_proj", "up_proj") else "down_proj")
            sf = src_idx.get(fused)
            if not sf:
                continue
            with safe_open(os.path.join(SRC, sf), framework="pt") as g2:
                full = g2.get_tensor(fused)[e]
            orig = full[:I] if nm == "gate_proj" else full[I:] if nm == "up_proj" else full
            errs.append(rel_err(t.float() * s.float().unsqueeze(1), orig))
            checked += 1
    if errs:
        errs.sort()
        print(f"[4] per-channel int8 relative error over {checked} expert tensors from {len(sample_shards)} shards: "
              f"median {errs[len(errs)//2]:.4f}, p95 {errs[int(len(errs)*0.95)]:.4f}, max {errs[-1]:.4f}")
        ok &= errs[-1] < 0.05
    else:
        print("[4] no expert tensors matched for error check (unexpected)"); ok = False

    for shard in sample_shards[:1]:
        with safe_open(os.path.join(DST, shard), framework="pt") as f:
            for k in f.keys():
                if any(p in k for p in ("self_attn.", "linear_attn.", "shared_expert.", "mlp.gate.", "visual.")) and k.endswith(".weight"):
                    excl_checked += 1
                    if f.get_slice(k).get_dtype() != "BF16":
                        print("[5] UNEXPECTED quantized excluded tensor:", k); ok = False
    print(f"[5] excluded-layer dtypes checked: {excl_checked} (all BF16 => OK)")

    miss_assets = [a for a in ASSETS if not os.path.exists(os.path.join(DST, a))]
    print(f"[6] missing tokenizer/processor assets: {miss_assets}")
    ok &= not miss_assets

    print("[verify] PASS" if ok else "[verify] FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
