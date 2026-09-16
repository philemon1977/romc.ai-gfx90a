#!/usr/bin/env python3
"""Benchmark the gfx90a flash-decoding kernel against the measured system wall.

The point is comparability, not microbench vanity numbers. On this box the real
server's decode TPOT is (single stream, official client, no speculation):

    ctx      TPOT      excess vs 1k   per full-attention layer (15 of 60 layers)
    1k       45.4 ms        --              --
    32k     171.1 ms       125.7 ms         8.38 ms
    128k    553.9 ms       508.5 ms        33.90 ms
    240k    967.9 ms       922.5 ms        61.50 ms

so a kernel is only interesting if it makes `45.4 + 15 * t_layer` collapse toward
45.4 ms. Each row below prints exactly that prediction, plus the effective KV
bandwidth so the roofline distance is visible (one GCD of MI250X: ~1.3 TB/s,
104 CUs).

Run alone (nothing else on the device) inside hyperloom-srv:
    HIP_VISIBLE_DEVICES=0 /opt/envs/vllm/bin/python3 bench_flash_decode.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import triton
from triton.testing import do_bench

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flash_decode import paged_decode_no_split, paged_flash_decode  # noqa: E402

DEV = "cuda"
HEAD_DIM = 256
BLOCK_SIZE = 528
N_KV = 1          # per TP rank for this checkpoint (num_key_value_heads=2, TP8)
N_PER_KV = 4      # 32 query heads / TP8
FULL_ATTN_LAYERS = 15
TPOT_1K_MS = 45.4           # measured whole-model decode step at 1k context
MEASURED = {1024: 45.4, 32768: 171.1, 131072: 553.9, 237568: 967.9}
PEAK_GBS = 1300.0           # ~1.3 TB/s per MI250X GCD


def make_tensors(ctx: int, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    blocks = max(1, math.ceil(ctx / BLOCK_SIZE))
    k = (torch.rand((blocks, BLOCK_SIZE, N_KV, HEAD_DIM), generator=g) * 2 - 1).bfloat16().to(DEV)
    v = (torch.rand((blocks, BLOCK_SIZE, N_KV, HEAD_DIM), generator=g) * 2 - 1).bfloat16().to(DEV)
    q = (torch.rand((1, N_KV * N_PER_KV, HEAD_DIM), generator=g) * 2 - 1).bfloat16().to(DEV)
    bt = torch.arange(blocks, dtype=torch.int32, device=DEV).view(1, blocks)
    cl = torch.tensor([ctx], dtype=torch.int32, device=DEV)
    return q, k, v, bt, cl


def kv_bytes(ctx: int) -> int:
    return 2 * ctx * N_KV * HEAD_DIM * 2  # K and V, bf16


def run(args) -> int:
    if not torch.cuda.is_available():
        print("no ROCm device visible")
        return 2
    name = torch.cuda.get_device_name(0)
    ncu = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"device = {name} | CUs = {ncu} | q heads/rank = {N_KV*N_PER_KV}, "
          f"kv heads/rank = {N_KV}, head_dim = {HEAD_DIM}, block = {BLOCK_SIZE}")
    print(f"target to beat: TPOT(ctx) = {TPOT_1K_MS} + 15 x t_layer  vs measured "
          f"{ {k: v for k, v in MEASURED.items()} } ms\n")

    ctxs = [int(x) for x in args.ctx.split(",")]
    for ctx in ctxs:
        q, k, v, bt, cl = make_tensors(ctx)
        row_bytes = kv_bytes(ctx)
        print(f"ctx={ctx}")
        print(f"  {'variant':<34} {'t_layer':>9} {'GB/s':>8} {'%peak':>7} {'pred TPOT':>10} {'tok/s':>7}")
        variants = []
        if args.include_nosplit:
            variants.append(
                ("no_split (1 workgroup baseline)",
                 lambda: paged_decode_no_split(q, k, v, bt, cl, num_kv_heads=N_KV,
                                               block_size=BLOCK_SIZE, block_n=args.block_n,
                                               num_warps=args.warps, num_stages=args.stages)))
        for ns in [int(x) for x in args.splits.split(",")]:
            variants.append((
                f"flash_decode num_splits={ns}",
                (lambda ns=ns: paged_flash_decode(q, k, v, bt, cl, num_kv_heads=N_KV,
                                                  block_size=BLOCK_SIZE, num_splits=ns,
                                                  block_n=args.block_n, num_warps=args.warps,
                                                  num_stages=args.stages))))
        for label, fn in variants:
            try:
                ms = do_bench(fn, warmup=args.warmup, rep=args.rep)
            except Exception as exc:  # noqa: BLE001 - a config that fails is a result row
                print(f"  {label:<34} FAILED {type(exc).__name__}: {str(exc)[:70]}")
                continue
            gbs = row_bytes / (ms * 1e-3) / 1e9
            pred = TPOT_1K_MS + FULL_ATTN_LAYERS * ms
            print(f"  {label:<34} {ms:>7.3f}ms {gbs:>7.1f} {gbs/PEAK_GBS*100:>6.2f}% "
                  f"{pred:>8.1f}ms {1000/pred:>7.2f}")
        cur = MEASURED.get(ctx)
        if cur:
            print(f"  current in-service per layer: {(cur - TPOT_1K_MS)/FULL_ATTN_LAYERS:.2f} ms "
                  f"(TPOT {cur} ms)")
        del q, k, v, bt, cl
        torch.cuda.empty_cache()
        if args.block_n_sweep:
            for bn in (32, 64, 128):
                for w in (4, 8):
                    for st in (1, 2, 3):
                        try:
                            ms = do_bench(lambda bn=bn, w=w, st=st: paged_flash_decode(
                                *make_tensors(min(ctx, 8192))[:1], *make_tensors(min(ctx, 8192))[1:],
                                num_kv_heads=N_KV, block_size=BLOCK_SIZE, block_n=bn,
                                num_warps=w, num_stages=st), warmup=5, rep=20)
                            print(f"    tune block_n={bn:>3} warps={w} stages={st} -> {ms:.3f} ms "
                                  f"(ctx {min(ctx,8192)})")
                        except Exception as exc:  # noqa: BLE001
                            print(f"    tune block_n={bn} warps={w} stages={st} -> {type(exc).__name__}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="1024,32768,131072,237568")
    ap.add_argument("--splits", default="8,32,64,104,128")
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--include-nosplit", action="store_true")
    ap.add_argument("--block-n-sweep", action="store_true", help="also sweep tile configs")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rep", type=int, default=50)
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
