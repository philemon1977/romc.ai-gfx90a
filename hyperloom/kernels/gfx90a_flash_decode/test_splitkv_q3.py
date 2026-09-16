#!/usr/bin/env python3
"""Correctness test for the q>1 (speculative-decoding) extension of split-KV.

Oracle: the UNMODIFIED in-tree module, called once per draft token with
``max_query_len=1`` and ``seq_lens = ctx - Q + 1 + tok`` -- which is exactly the
causal visibility of token ``tok`` inside a spec-decode step, and is code we trust
(it agrees bit-for-bit with the serial kernel today).

Under test: ``draft_splitkv_q3.py`` called ONCE with a ``[Q, heads, dim]`` query.

If the row mapping ((tok, q-head) pairs into the padded M tile), the scratch axis
(``heads * Q``), the per-row causal bound and the reduce grid are all right, the
two agree within bf16 noise -- and the whole point of the patch is that they do it
while the KV is read **once** instead of Q times.

    VLLM_ROCM_SPLITKV_PA=1 HIP_VISIBLE_DEVICES=0 python3 test_splitkv_q3.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OVERLAY = Path("/home/qiba/ROCm.AI/hyperloom/patches/fp8-w8a8-emulation-gfx90a")
sys.path.insert(0, str(OVERLAY))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

HEAD_DIM, N_KV, N_PER_KV, X = 256, 1, 4, 8
Q = 3                      # num_speculative_tokens=2 -> 3 rows per verify step
PHYS = 528                 # hybrid attention block size for this checkpoint
TABLE_COLS = 497           # max_model_len / 528: what a live server hands over
CTXS = [4096, 32768, 131072]


def load_draft():
    spec = importlib.util.spec_from_file_location("draft_splitkv_q3", HERE / "draft_splitkv_q3.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["draft_splitkv_q3"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_step(ctx: int, q_tokens: int, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    blocks = max(1, math.ceil(ctx / PHYS))
    assert TABLE_COLS >= blocks, "table must cover the sequence"
    pool = TABLE_COLS + 8
    kk = (torch.rand((pool, N_KV, HEAD_DIM // X, PHYS, X), generator=g) * 2 - 1)
    vv = (torch.rand((pool, N_KV, HEAD_DIM // X, PHYS, X), generator=g) * 2 - 1)
    q = (torch.rand((q_tokens, N_KV * N_PER_KV, HEAD_DIM), generator=g) * 2 - 1) * 2.5
    return {
        "key_cache": kk.bfloat16().cuda().contiguous(),
        "value_cache": vv.bfloat16().cuda().contiguous(),
        "query": q.bfloat16().cuda().contiguous(),
        "block_table": torch.arange(TABLE_COLS, dtype=torch.int32).view(1, -1).cuda(),
        "ctx": ctx,
    }


def run_q1(mod, t, tok: int, q_rows_total: int):
    """One draft token through the plain (max_query_len==1) path."""
    # seq_len as seen by token `tok`: everything up to and including its own slot
    vis = t["ctx"] - q_rows_total + 1 + tok
    q = t["query"][tok:tok + 1].contiguous()
    out = torch.empty_like(q)
    ok = mod.try_paged_decode(
        query=q, output=out, key_cache=t["key_cache"], value_cache=t["value_cache"],
        block_table=t["block_table"], query_start_loc=torch.tensor([0, 1], dtype=torch.int32,
                                                                   device="cuda"),
        seq_lens=torch.tensor([vis], dtype=torch.int32, device="cuda"),
        max_seq_len=TABLE_COLS * PHYS, max_query_len=1, kv_cache_dtype="auto",
        sliding_window=None, alibi_slopes=None, sinks=None,
        sm_scale=1.0 / math.sqrt(HEAD_DIM))
    return ok, out.squeeze(0).float()


def run_q3(draft, t):
    out = torch.empty_like(t["query"])
    ok = draft.try_paged_decode(
        query=t["query"], output=out, key_cache=t["key_cache"], value_cache=t["value_cache"],
        block_table=t["block_table"],
        query_start_loc=torch.tensor([0, Q], dtype=torch.int32, device="cuda"),
        seq_lens=torch.tensor([t["ctx"]], dtype=torch.int32, device="cuda"),
        max_seq_len=TABLE_COLS * PHYS, max_query_len=Q, kv_cache_dtype="auto",
        sliding_window=None, alibi_slopes=None, sinks=None,
        sm_scale=1.0 / math.sqrt(HEAD_DIM))
    return ok, out.float(), draft.stats()


def main() -> int:
    os.environ.setdefault("VLLM_ROCM_SPLITKV_PA", "1")
    os.environ["VLLM_ROCM_SPLITKV_PA_MAX_Q"] = str(Q)
    from vllm.v1.attention.ops import rocm_splitkv_pa as orig  # noqa: E402
    draft = load_draft()
    print(f"oracle = unmodified module, {Q} calls at max_query_len=1")
    print(f"under test = draft_splitkv_q3, 1 call at max_query_len={Q} "
          f"(draft module gate max_q={os.environ['VLLM_ROCM_SPLITKV_PA_MAX_Q']})\n")
    bad = 0
    for ctx in CTXS:
        t = make_step(ctx, Q)
        ref = []
        for tok in range(Q):
            ok, o = run_q1(orig, t, tok, Q)
            if not ok:
                print(f"  ctx={ctx}: ORACLE REJECTED at tok{tok} -> {orig.stats()['reject_by_reason']}")
                return 2
            ref.append(o)
        ref = torch.stack(ref)                      # [Q, heads, dim]
        ok3, got, st = run_q3(draft, t)
        if not ok3:
            print(f"  ctx={ctx}: DRAFT REJECTED -> {list((st.get('reject_by_reason') or {}).items())[-2:]}")
            bad += 1
            continue
        d = (got - ref).abs()
        signal = max(float(ref.abs().max()), 1e-6)
        max_abs = float(d.max())
        rel = max_abs / signal
        per_tok = [float(x) for x in d.amax(dim=(1, 2))]
        okp = max_abs <= 2e-2 or rel <= 2e-2
        ll = st.get("last_launch") or {}
        print(f"  ctx={ctx:<7} {'PASS' if okp else 'FAIL'}  max_abs={max_abs:.4f} "
              f"signal={signal:.3f} rel={rel*100:.2f}%  per_tok_max={['%.4f' % x for x in per_tok]} "
              f"parts={ll.get('max_parts')}")
        bad += 0 if okp else 1
        del t, ref, got
        torch.cuda.empty_cache()
    print("\n" + ("Q>1 EXTENSION MATCHES THE ORACLE" if bad == 0 else f"{bad} MISMATCHES"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
