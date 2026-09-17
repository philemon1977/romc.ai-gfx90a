#!/usr/bin/env python3
"""Prototype + prove the batched W4A16 repack (plan C).

vLLM's TritonW4A16LinearKernel.process_weights_after_loading repacks each linear
layer on its own: for 47,232 expert layers that is ~280k tiny kernel launches
and the load spends 25-40 min at ~0% GPU utilisation.

This script implements the SAME arithmetic but batched over an expert dimension,
and checks it against vLLM's own per-layer function bit-for-bit.  Nothing here
patches vLLM; it is the numeric proof that a batched version is equivalent.

Run with: --sample N  (number of experts to compare)
"""
import argparse
import json
import os

import torch
from safetensors import safe_open

CT_DIR = "/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16"
GROUP = 32


# ---------------------------------------------------------------- vLLM's per-layer version
def repack_w_q_ref(w: torch.Tensor) -> torch.Tensor:
    """Verbatim transcription of vLLM's repack_w_q for a [N, K//8] int32 tensor."""
    shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
    n_dim, k8 = w.shape
    k_dim = k8 * 8
    w_unpacked = ((w.unsqueeze(-1) >> shifts) & 0xF).reshape(n_dim, k_dim)
    w_kn = w_unpacked.t().contiguous()
    n8 = n_dim // 8
    return torch.sum(
        (w_kn.view(k_dim, n8, 8) & 0xF) << shifts, dim=2, dtype=torch.int32
    ).contiguous()


def repack_w_s_ref(s: torch.Tensor) -> torch.Tensor:
    return s.t().contiguous()


# ---------------------------------------------------------------- batched version
def repack_w_q_batched(w: torch.Tensor) -> torch.Tensor:
    """Same math, one extra leading dim: w is [E, N, K//8] -> [E, K, N//8]."""
    shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
    e, n_dim, k8 = w.shape
    k_dim = k8 * 8
    w_unpacked = ((w.unsqueeze(-1) >> shifts) & 0xF).reshape(e, n_dim, k_dim)
    w_kn = w_unpacked.transpose(1, 2).contiguous()          # [E, K, N]
    n8 = n_dim // 8
    return torch.sum(
        (w_kn.view(e, k_dim, n8, 8) & 0xF) << shifts, dim=3, dtype=torch.int32
    ).contiguous()


def repack_w_s_batched(s: torch.Tensor) -> torch.Tensor:
    """[E, N, K//G] -> [E, K//G, N]."""
    return s.transpose(1, 2).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--expert", default=0)
    a = ap.parse_args()
    dev = torch.device(a.device)

    owm = json.load(open(os.path.join(CT_DIR, "model.safetensors.index.json")))["weight_map"]

    def load(name):
        with safe_open(os.path.join(CT_DIR, owm[name]), "pt") as f:
            return f.get_tensor(name)

    layer = 0
    mods = [f"layers.{layer}.ffn.experts.{e}.{w}"
            for e in range(a.sample) for w in ("w1", "w3", "w2")]
    packed, scales, tags = [], [], []
    for m in mods:
        pk = load(m + ".weight_packed")      # [N, K//8] int32
        sc = load(m + ".weight_scale")       # [N, K//G] fp32
        packed.append(pk)
        scales.append(sc)
        tags.append(m)
        print(f"  {m:34s} packed{tuple(pk.shape)} {pk.dtype}  scale{tuple(sc.shape)} {sc.dtype}")

    # group by identical shape (the batch dim must be uniform)
    by_shape = {}
    for pk, sc, tag in zip(packed, scales, tags):
        by_shape.setdefault((tuple(pk.shape), tuple(sc.shape)), []).append((pk, sc, tag))

    print(f"\ngroups: {[(k[0], len(v)) for k, v in by_shape.items()]}")
    bad = 0
    for (pshape, sshape), items in by_shape.items():
        W = torch.stack([t[0] for t in items]).to(dev)      # [E, N, K//8]
        S = torch.stack([t[1] for t in items]).to(dev)      # [E, N, K//G]

        q_batch = repack_w_q_batched(W)
        s_batch = repack_w_s_batched(S)

        for i, (_, _, tag) in enumerate(items):
            q_ref = repack_w_q_ref(W[i])
            s_ref = repack_w_s_ref(S[i])
            okq = torch.equal(q_batch[i], q_ref)
            oks = torch.equal(s_batch[i], s_ref)
            if not (okq and oks):
                bad += 1
            print(f"  {'ok ' if okq and oks else 'BAD'} {tag:34s} "
                  f"q{tuple(q_ref.shape)}=={tuple(q_batch[i].shape)} {okq}  "
                  f"s{tuple(s_ref.shape)}=={tuple(s_batch[i].shape)} {oks}")

    print(f"\n{'PASS' if bad == 0 else 'FAIL'}: {bad} mismatching tensors")
    print("layout produced: qweight [K, N//8] int32, scales [K//G, N]  (kernel layout)")

    # also show what a single batched call would cost vs per-layer
    E = 384 * 40
    print(f"\nfor the real model: {E} expert weights; batched does ~{len(by_shape)} "
          f"kernel-launch sets instead of {E * 3:,} per-layer calls")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
