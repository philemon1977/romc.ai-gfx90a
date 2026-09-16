#!/usr/bin/env python3
"""Correctness harness for the gfx90a flash-decoding kernel.

Discipline copied from AMD's MI250/gfx90a kernel-optimization case study
(https://rocm.blogs.amd.com/software-tools-optimization/profiling-guide/ai-assist-optimization/README.html):
one fixed reference, a numeric diff as the ONLY gate for keeping a change, and
every case recorded. The reference here is fp32 attention over *gathered* (unpaged)
KV, so it shares no code path with the kernel under test.

Deliberately hostile cases:
  * ctx < block, == block, == block+1 (partial tail), 0-length-safe splits
  * a scrambled block table, so an accidental identity-map cannot pass
  * GQA group 4 (the real TP8 shape: 4 q heads per rank per KV head) and group 8
    with 2 KV heads (what an unsharded run looks like)
  * ctx up to 131,072 at fp32-reference cost of ~134 MB per KV tensor

Run inside hyperloom-srv on one die:
    HIP_VISIBLE_DEVICES=0 /opt/envs/vllm/bin/python3 test_flash_decode.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flash_decode import paged_decode_no_split, paged_flash_decode  # noqa: E402

DEV = "cuda"
HEAD_DIM = 256
BLOCK_SIZE = 528  # vLLM's hybrid attention block size for this checkpoint
# bf16 inputs with fp32 accumulation vs an fp32 reference: these bounds are what
# the dtype affords, not what the kernel happens to achieve.
ATOL, RTOL = 2e-2, 2e-2


def make_case(bsz: int, n_kv: int, n_per_kv: int, ctx: int, *, blocks_extra: int = 4,
              seed: int = 0):
    """Paged bf16 KV cache with a scrambled block table + fp32 gathered reference."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_q = n_kv * n_per_kv
    blocks_per_seq = max(1, math.ceil(ctx / BLOCK_SIZE))
    total_blocks = bsz * (blocks_per_seq + blocks_extra)

    k_pool = torch.rand((total_blocks, BLOCK_SIZE, n_kv, HEAD_DIM), generator=g) * 2 - 1
    v_pool = torch.rand((total_blocks, BLOCK_SIZE, n_kv, HEAD_DIM), generator=g) * 2 - 1
    # scaled so the softmax is peaked (real decode attention); with flat q the
    # output is an average of zero-mean vectors and an absolute tolerance is noise
    q_f = (torch.rand((bsz, n_q, HEAD_DIM), generator=g) * 2 - 1) * 2.5

    k_cache, v_cache = k_pool.bfloat16().to(DEV), v_pool.bfloat16().to(DEV)
    q = q_f.bfloat16().to(DEV)

    table = torch.zeros((bsz, blocks_per_seq), dtype=torch.long)
    ref_k = torch.zeros((bsz, ctx, n_kv, HEAD_DIM), dtype=torch.float32)
    ref_v = torch.zeros((bsz, ctx, n_kv, HEAD_DIM), dtype=torch.float32)
    for b in range(bsz):
        ids = torch.randperm(total_blocks, generator=g)[:blocks_per_seq]
        table[b] = ids
        for j, blk in enumerate(ids.tolist()):
            lo, hi = j * BLOCK_SIZE, min((j + 1) * BLOCK_SIZE, ctx)
            if lo >= ctx:
                break
            ref_k[b, lo:hi] = k_pool[blk, : hi - lo]
            ref_v[b, lo:hi] = v_pool[blk, : hi - lo]

    ctx_lens = torch.full((bsz,), ctx, dtype=torch.int32, device=DEV)

    kk = ref_k.to(DEV).permute(0, 2, 1, 3)                       # [b, kv, ctx, d]
    vv = ref_v.to(DEV).permute(0, 2, 1, 3)
    qq = q.float().reshape(bsz, n_kv, n_per_kv, HEAD_DIM)        # [b, kv, grp, d]
    scale = 1.0 / math.sqrt(HEAD_DIM)
    logits = torch.einsum("bgtd,bgcd->bgct", qq, kk) * scale
    probs = torch.softmax(logits, dim=-1)
    ref = torch.einsum("bgct,bgcd->bgtd", probs, vv).reshape(bsz, n_q, HEAD_DIM)
    return q, k_cache, v_cache, table.to(DEV).to(torch.int32), ctx_lens, n_kv, ref


def check(name: str, got: torch.Tensor, ref: torch.Tensor, *, atol: float = ATOL,
          rtol: float = RTOL) -> bool:
    d = (got.float() - ref.float()).abs()
    max_abs = float(d.max())
    signal = max(float(ref.float().abs().max()), 1e-6)
    max_rel = max_abs / signal                     # relative to the signal scale
    ok = (max_abs <= atol) or (max_rel <= rtol)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<36} max_abs={max_abs:.4f} max_rel={max_rel:.4f}")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("no ROCm device visible")
        return 2
    print(f"device = {torch.cuda.get_device_name(0)} | paged bf16 KV, block={BLOCK_SIZE}, "
          f"head_dim={HEAD_DIM}, fp32 reference on gathered KV")
    cases = [
        (1, 1, 4, 1), (1, 1, 4, 5), (1, 1, 4, 527), (1, 1, 4, 528), (1, 1, 4, 529),
        (1, 1, 4, 1024), (2, 1, 4, 4097), (1, 2, 4, 1000), (1, 1, 8, 2048),
        (1, 1, 4, 32768), (1, 1, 4, 131072),
    ]
    bad = 0
    for bsz, n_kv, n_per_kv, ctx in cases:
        q, kc, vc, bt, cl, nk, ref = make_case(bsz, n_kv, n_per_kv, ctx)
        tag = f"b{bsz} kv{n_kv} grp{n_per_kv} ctx{ctx}"
        flash = paged_flash_decode(q, kc, vc, bt, cl, num_kv_heads=nk, block_size=BLOCK_SIZE)
        bad += 0 if check(f"flash_decode {tag}", flash, ref) else 1
        if ctx <= 8192:  # the unsplit baseline is O(ctx) serial: only cheap cases
            nos = paged_decode_no_split(q, kc, vc, bt, cl, num_kv_heads=nk, block_size=BLOCK_SIZE)
            bad += 0 if check(f"nosplit      {tag}", nos, ref) else 1
            bad += 0 if check(f"flash==nosplit {tag}", flash, nos, atol=3e-2, rtol=3e-2) else 1
        del q, kc, vc, bt, cl, ref, flash
        torch.cuda.empty_cache()

    q, kc, vc, bt, cl, nk, ref = make_case(1, 1, 4, 4097)
    one = paged_flash_decode(q, kc, vc, bt, cl, num_kv_heads=nk, block_size=BLOCK_SIZE, num_splits=1)
    many = paged_flash_decode(q, kc, vc, bt, cl, num_kv_heads=nk, block_size=BLOCK_SIZE, num_splits=104)
    bad += 0 if check("num_splits=1 ctx4097", one, ref) else 1
    bad += 0 if check("num_splits=104 ctx4097", many, ref) else 1
    agree = torch.allclose(one.float(), many.float(), atol=3e-2, rtol=3e-2)
    print(f"  {'PASS' if agree else 'FAIL'}  split-count invariance                 "
          f"max_abs={float((one.float()-many.float()).abs().max()):.4f}")
    bad += 0 if agree else 1
    print("\n" + ("ALL PASS" if bad == 0 else f"{bad} FAILURES"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
