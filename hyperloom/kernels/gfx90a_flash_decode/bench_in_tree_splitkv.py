#!/usr/bin/env python3
"""Verify and benchmark the IN-TREE gfx90a split-KV kernel without booting vLLM.

Context: this wheel already carries a MI250X split-KV paged-decode implementation
at ``vllm/v1/attention/ops/rocm_splitkv_pa.py`` (dispatched from
``chunked_prefill_paged_decode.py`` right before the serial single-CTA fallback),
gated OFF by default behind ``VLLM_ROCM_SPLITKV_PA=1``. Its header cites
"576/576 cells pass an independent float32 golden judge" and "fixed PART=64: worst
189 GB/s / x28.2 over 12 shapes" -- but the harness it cites
(``bench/pa_splitkv.py``, ``pa256_golden.py``) is NOT on this machine, and there is
no tests/ dir. So the claims are unaudited here.

This script is that audit, without a 12-minute server boot: it builds the exact
5-D packed KV layout the gate demands, calls ``try_paged_decode`` directly, and
compares against
  1. an fp32 reference computed on gathered KV (shares no code with either kernel),
  2. the serial fallback ``kernel_paged_attention_2d`` (what runs today),
  3. our own independent split-KV implementation (flash_decode.py), which doubles
     as a second opinion if the in-tree one and the reference disagree.

Then it times all three and reports GB/s and the implied whole-model TPOT
(``45.4 ms + 15 x t_layer``), which is the number the live server is judged on
(measured today: 553.9 ms at 128k, i.e. 33.9 ms per layer).

Run inside hyperloom-srv on an otherwise idle die:
    VLLM_ROCM_SPLITKV_PA=1 HIP_VISIBLE_DEVICES=0 \
      /opt/envs/vllm/bin/python3 bench_in_tree_splitkv.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_ROCM_SPLITKV_PA", "1")

import torch  # noqa: E402
from triton.testing import do_bench  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/home/qiba/ROCm.AI/hyperloom/patches/"
                            "fp8-w8a8-emulation-gfx90a")))

from flash_decode import paged_flash_decode  # noqa: E402

HEAD_DIM = 256
N_KV = 1
N_PER_KV = 4
LAYERS = 15
TPOT_1K = 45.4
MEASURED_TODAY = {1024: 45.4, 32768: 171.1, 131072: 553.9, 237568: 967.9}
X = 16 // torch.tensor(0, dtype=torch.bfloat16).element_size()  # bf16 -> 8 elems


def _to_logical(t: torch.Tensor) -> torch.Tensor:
    """[blocks, kv, dim//x, phys_block, x] -> [blocks, phys_block, kv, dim]."""
    blocks, _, d_over_x, phys, x = t.shape
    return t.permute(0, 3, 1, 2, 4).reshape(blocks, phys, N_KV, d_over_x * x).contiguous()


def build(ctx: int, phys_block: int, seed: int = 0):
    """Paged KV in the layout the gate checks: [blocks, kv_heads, dim//x, block, x]."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    blocks = max(1, math.ceil(ctx / phys_block))
    # A live server hands the kernel a block_table whose COLUMN COUNT is sized for
    # max_model_len, not for this request (the static bound the split kernel grids
    # over is columns x phys_block). PROBE_TABLE_COLS reproduces that; without it a
    # minimal table can turn a harmless-to-the-server read into a fault in here.
    cols = max(blocks, int(os.environ.get("PROBE_TABLE_COLS", "0")))
    pool_blocks = cols + 8
    kk = (torch.rand((pool_blocks, N_KV, HEAD_DIM // X, phys_block, X), generator=g) * 2 - 1)
    vv = (torch.rand((pool_blocks, N_KV, HEAD_DIM // X, phys_block, X), generator=g) * 2 - 1)
    # q scaled so the softmax is PEAKED: real decode attention is not uniform over
    # 128k tokens, and with flat q the output is a mean of zero-mean vectors (~0.01),
    # which makes any absolute tolerance meaningless.
    q = (torch.rand((1, N_KV * N_PER_KV, HEAD_DIM), generator=g) * 2 - 1) * 2.5

    # de-interleaved fp32 reference view: [blocks, kv, dim//x, block, x] ->
    # [blocks, block, kv, dim]
    kl, vl = _to_logical(kk).float(), _to_logical(vv).float()
    gathered_k = kl[: math.ceil(ctx / phys_block)].reshape(-1, N_KV, HEAD_DIM)[:ctx]
    gathered_v = vl[: math.ceil(ctx / phys_block)].reshape(-1, N_KV, HEAD_DIM)[:ctx]

    logits = (q.float().squeeze(0).reshape(N_PER_KV, HEAD_DIM) @ gathered_k.squeeze(1).T) \
        / math.sqrt(HEAD_DIM)
    ref = torch.softmax(logits, dim=-1) @ gathered_v.squeeze(1)   # [n_per_kv, dim]

    table = torch.arange(cols, dtype=torch.int32).view(1, cols)
    return {
        "query": q.bfloat16().cuda(),
        "key_cache": kk.bfloat16().cuda().contiguous(),
        "value_cache": vv.bfloat16().cuda().contiguous(),
        "block_table": table.cuda(),
        "seq_lens": torch.tensor([ctx], dtype=torch.int32, device="cuda"),
        "query_start_loc": torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        "phys_block": phys_block,
        "ref": ref.float().cuda(),
        "ctx": ctx,
    }


def err_report(name: str, got: torch.Tensor, ref: torch.Tensor) -> tuple[bool, float, float]:
    """PASS if the error is small in absolute terms OR relative to the signal."""
    d = (got.float() - ref.float()).abs()
    max_abs = float(d.max())
    signal = float(ref.float().abs().max())
    rel = max_abs / max(signal, 1e-6)
    ok = max_abs <= 2e-2 or rel <= 2e-2
    print(f"  {name:<28} max_abs={max_abs:.4f}  signal={signal:.3f}  rel={rel*100:.2f}%  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok, max_abs, rel


def call_in_tree(t, part: int | None = None):
    """Let the in-tree kernel take over; returns (ok, out)."""
    from vllm.v1.attention.ops import rocm_splitkv_pa as sk

    if part is not None:
        os.environ["VLLM_ROCM_SPLITKV_PA_PART"] = str(part)
    out = torch.empty_like(t["query"])
    ok = sk.try_paged_decode(
        query=t["query"], output=out, key_cache=t["key_cache"], value_cache=t["value_cache"],
        block_table=t["block_table"], query_start_loc=t["query_start_loc"],
        seq_lens=t["seq_lens"], max_seq_len=t["ctx"], max_query_len=1,
        kv_cache_dtype="auto", sliding_window=None, alibi_slopes=None, sinks=None,
        sm_scale=1.0 / math.sqrt(HEAD_DIM),
    )
    return ok, out, sk.stats()


def call_serial(t):
    """The path that runs today: chunked_prefill_paged_decode -> the serial
    ``kernel_paged_attention_2d`` with grid (num_seqs, num_kv_heads) = (1, 1)."""
    from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
        chunked_prefill_paged_decode,
    )

    out = torch.empty_like(t["query"])
    # signature: query, key, value, output, kv_cache_dtype, key_cache, value_cache,
    #            block_table, query_start_loc, seq_lens, max_seq_len, max_query_len,
    #            k_scale, v_scale, ...   (key/value are unused when max_query_len==1)
    # the wrapper itself tries split-KV first, so force the REAL fallback off
    prev = os.environ.get("VLLM_ROCM_SPLITKV_PA", "0")
    os.environ["VLLM_ROCM_SPLITKV_PA"] = "0"
    try:
        _serial_call(t, out)
    finally:
        os.environ["VLLM_ROCM_SPLITKV_PA"] = prev
    return out


def _serial_call(t, out):
    from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
        chunked_prefill_paged_decode,
    )

    chunked_prefill_paged_decode(
        t["query"], None, None, out, "auto",
        t["key_cache"], t["value_cache"], t["block_table"],
        t["query_start_loc"], t["seq_lens"], t["ctx"], 1,
        None, None, sm_scale=1.0 / math.sqrt(HEAD_DIM),
    )
    return out


def main() -> int:
    if not torch.cuda.is_available():
        print("no ROCm device")
        return 2
    print(f"device = {torch.cuda.get_device_name(0)}")
    import vllm
    from vllm.v1.attention.ops import rocm_splitkv_pa as sk

    print(f"vllm {vllm.__version__} | split-KV enabled()={sk.enabled()}")
    print(f"gate: ENV VLLM_ROCM_SPLITKV_PA={os.environ.get('VLLM_ROCM_SPLITKV_PA')} "
          f"PART={os.environ.get('VLLM_ROCM_SPLITKV_PA_PART', 'adaptive')} "
          f"MAX_SEQS={os.environ.get('VLLM_ROCM_SPLITKV_PA_MAX_SEQS', 'default')}\n")

    ctxs = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
                             else ["4096", "131072"])]
    bad = 0
    for ctx in ctxs:
        for phys_block in (528,):
            print(f"--- ctx={ctx} phys_block={phys_block} ---")
            t = build(ctx, phys_block)
            ok, out_sk, st = call_in_tree(t)
            if not ok:
                print(f"  NOT TAKEN OVER: reject_by_reason={st.get('reject_by_reason')}")
                bad += 1
                continue
            ll = {k: v for k, v in (st.get("last_launch") or {}).items()}
            print(f"  takeover OK  last_launch={ll}")
            ok_num, _, _ = err_report("in-tree vs fp32 ref", out_sk.float().squeeze(0), t["ref"])
            bad += 0 if ok_num else 1

            ms_sk = do_bench(lambda: call_in_tree(t)[1], warmup=10, rep=50)
            gb = 2 * ctx * N_KV * HEAD_DIM * 2 / (ms_sk * 1e-3) / 1e9
            print(f"  in-tree split-KV : {ms_sk*1000:>8.1f} us/layer  {gb:>7.1f} GB/s  "
                  f"implied TPOT {TPOT_1K + LAYERS*ms_sk:>8.1f} ms")

            try:
                out_ser = call_serial(t)
                err_report("serial vs fp32 ref", out_ser.float().squeeze(0), t["ref"])
                ms_ser = do_bench(lambda: call_serial(t), warmup=3, rep=10)
                gbs = 2 * ctx * N_KV * HEAD_DIM * 2 / (ms_ser * 1e-3) / 1e9
                print(f"  serial fallback  : {ms_ser*1000:>8.1f} us/layer  {gbs:>7.1f} GB/s  "
                      f"implied TPOT {TPOT_1K + LAYERS*ms_ser:>8.1f} ms  "
                      f"speedup x{ms_ser/ms_sk:.2f}")
            except Exception as exc:  # noqa: BLE001
                print(f"  serial fallback  : could not invoke ({type(exc).__name__}: {str(exc)[:70]})")

            kl = _to_logical(t["key_cache"])
            vl = _to_logical(t["value_cache"])
            o2 = paged_flash_decode(t["query"], kl, vl, t["block_table"].long(),
                                    t["seq_lens"].long(), num_kv_heads=N_KV,
                                    block_size=phys_block)
            dc = (o2.float().squeeze(0) - out_sk.float().squeeze(0)).abs()
            print(f"  our flash_decode vs in-tree: max_abs={float(dc.max()):.4f} "
                  f"{'AGREE' if float(dc.max()) <= 3e-2 else 'DISAGREE'}")
            if ctx in MEASURED_TODAY:
                print(f"  live server today at this ctx: TPOT {MEASURED_TODAY[ctx]} ms")
            del t, out_sk
            torch.cuda.empty_cache()
    print("\n" + ("AUDIT OK" if bad == 0 else f"{bad} PROBLEMS"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
