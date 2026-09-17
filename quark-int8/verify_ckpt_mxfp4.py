#!/usr/bin/env python3
"""Verify the MXFP4 (W4A16) Quark checkpoint: structure + real numerical
round-trip (unpack fp4 + decode e8m0 scales, compare with the bf16 source)."""
import json
import os
import random
import sys

import torch
from safetensors import safe_open

SRC = "/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B"
DST = os.environ.get("DST", "/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-Quark-MXFP4-W4A16")
NL, NE, H, I = 60, 512, 4096, 1024
GROUP = 32

# fp4 (E2M1) codebook: index -> value, sign in the high nibble
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def unpack_fp4(packed: torch.Tensor, out_features: int) -> torch.Tensor:
    """packed uint8 [N, K/2] -> float [N, K] (low nibble = first element)."""
    lo = (packed & 0x0F).to(torch.int64)
    hi = ((packed >> 4) & 0x0F).to(torch.int64)
    vals = torch.empty(packed.shape[0], packed.shape[1] * 2, dtype=torch.float32)
    vals[:, 0::2] = E2M1[lo]
    vals[:, 1::2] = E2M1[hi]
    sign = torch.where((packed & 0x08) != 0, -1.0, 1.0)
    return vals


def decode_e8m0(u8: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, u8.to(torch.float32) - 127.0)


def dequant_mxfp4(w_u8: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    """[N, K/2] uint8 + [N, K/32] e8m0 -> [N, K] float32."""
    N, Khalf = w_u8.shape
    K = Khalf * 2
    q = torch.empty(N, K, dtype=torch.float32)
    for i in range(K // 2):  # nibble order handled per byte
        b = w_u8[:, i].to(torch.int64)
        lo = b & 0x0F
        hi = (b >> 4) & 0x0F
        q[:, 2 * i] = torch.where((lo & 0x08) != 0, -E2M1[lo & 0x07], E2M1[lo & 0x07])
        q[:, 2 * i + 1] = torch.where((hi & 0x08) != 0, -E2M1[hi & 0x07], E2M1[hi & 0x07])
    s = decode_e8m0(scale_u8).repeat_interleave(GROUP, dim=1)
    return q * s


def main() -> None:
    ok = True
    cfg = json.load(open(os.path.join(DST, "config.json")))
    src_cfg = json.load(open(os.path.join(SRC, "config.json")))
    assert cfg["architectures"] == src_cfg["architectures"]
    g = cfg["quantization_config"]["global_quant_config"]
    w = g["weight"]
    print(f"[1] weight config: dtype={w['dtype']} qscheme={w['qscheme']} group_size={w['group_size']} "
          f"scale_format={w.get('scale_format')} dynamic={w['is_dynamic']} | input={g.get('input_tensors')}")
    ok &= w["dtype"] == "fp4" and w["qscheme"] == "per_group" and w["group_size"] == GROUP
    ok &= g.get("input_tensors") is None

    idx = json.load(open(os.path.join(DST, "model.safetensors.index.json")))["weight_map"]
    files = sorted(set(idx.values()))
    missing = [f for f in files if not os.path.exists(os.path.join(DST, f))]
    total = sum(os.path.getsize(os.path.join(DST, f)) for f in files if os.path.exists(os.path.join(DST, f)))
    print(f"[2] index: {len(idx)} tensors / {len(files)} shards, missing={len(missing)}, total {total/1e9:.1f} GB")
    ok &= not missing

    n_w = n_s = 0
    for lyr in range(NL):
        for e in range(NE):
            for nm in ("gate_proj", "up_proj", "down_proj"):
                k = f"model.language_model.layers.{lyr}.mlp.experts.{e}.{nm}.weight"
                n_w += k in idx
                n_s += (k + "_scale") in idx
    exp = NL * NE * 3
    print(f"[3] expert tensors {n_w}/{exp}, scales {n_s}/{exp}")
    ok &= n_w == exp and n_s == exp

    src_idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
    random.seed(7)
    errs, checked = [], 0
    for shard in random.sample(files, min(3, len(files))):
        with safe_open(os.path.join(DST, shard), framework="pt") as f:
            keys = [k for k in f.keys() if k.endswith(".weight") and "experts" in k]
            if not keys:
                continue
            k = sorted(keys)[0]
            if f.get_slice(k).get_dtype() != "U8":
                print("[4] unexpected weight dtype for", k, f.get_slice(k).get_dtype()); ok = False; continue
            w_u8 = f.get_tensor(k)
            s_u8 = f.get_tensor(k + "_scale")
            m = k.split(".")
            lyr = int(m[m.index("layers") + 1]); e = int(m[m.index("experts") + 1]); nm = m[-2]
            fused = f"model.language_model.layers.{lyr}.mlp.experts." + ("gate_up_proj" if nm != "down_proj" else "down_proj")
            with safe_open(os.path.join(SRC, src_idx[fused]), framework="pt") as g2:
                full = g2.get_tensor(fused)[e]
            orig = full[:I] if nm == "gate_proj" else full[I:] if nm == "up_proj" else full
            recon = dequant_mxfp4(w_u8, s_u8)
            o = orig.float()
            rmse = ((recon - o).pow(2).mean().sqrt() / o.pow(2).mean().sqrt()).item()
            cos = torch.nn.functional.cosine_similarity(recon.flatten(), o.flatten(), dim=0).item()
            errs.append((rmse, cos)); checked += 1
            print(f"    {k}: weight {tuple(w_u8.shape)} scale {tuple(s_u8.shape)} "
                  f"normRMSE {rmse:.4f} cosine {cos:.5f}")
    if errs:
        worst_rmse = max(e[0] for e in errs)
        worst_cos = min(e[1] for e in errs)
        print(f"[4] MXFP4 round-trip over {checked} expert tensors: worst normRMSE {worst_rmse:.4f}, "
              f"worst cosine {worst_cos:.5f} (E2M1 = 3 mantissa bits; max-abs-error is NOT the right metric)")
        ok &= worst_rmse < 0.20 and worst_cos > 0.99
    else:
        print("[4] no expert tensors sampled"); ok = False

    with safe_open(os.path.join(DST, files[0]), framework="pt") as f:
        bad = [k for k in f.keys() if "self_attn" in k or "linear_attn" in k or "mlp.gate" in k
               if k.endswith(".weight") and f.get_slice(k).get_dtype() != "BF16"]
    print(f"[5] excluded-layer non-BF16 tensors in shard0: {len(bad)}")
    ok &= not bad

    print("[verify-mxfp4]", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
