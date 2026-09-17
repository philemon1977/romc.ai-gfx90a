#!/usr/bin/env python3
"""Validate the core cost assumption of an expert slot-cache without touching vLLM:

  - reserve one layer's MoE weight slots in HBM (TP8 layout: w13 [512,256,4096] int8
    + w2 [512,4096,128] int8, ~780 MB),
  - keep a pinned host staging buffer holding a cold expert (1.57 MB/rank),
  - measure the *end-to-end latency* of streaming one expert into its slot
    (the on-miss path): pinned H2D copy + event sync, repeated N times.

If this stays in the tens of microseconds, a 25%% cold-expert offload costs only a
few percent of decode time even with fully synchronous fetching.
"""
import statistics
import time

import torch

H, I, E = 4096, 1024 // 8, 512          # per-rank shard at TP8
W13_ROW = 2 * I                          # 256 rows per expert
DT = torch.int8


def main() -> None:
    dev = "cuda"
    # slot table for one layer (only a few slots are ever touched in this test)
    w13 = torch.zeros(E, W13_ROW, H, dtype=DT, device=dev)   # 512 MB
    w2 = torch.zeros(E, H, I, dtype=DT, device=dev)          # 256 MB
    scale13 = torch.zeros(E, W13_ROW, dtype=torch.float32, device=dev)
    scale2 = torch.zeros(E, H, dtype=torch.float32, device=dev)

    # pinned host copy of 8 "cold experts"
    host13 = torch.randint(-127, 128, (8, W13_ROW, H), dtype=DT, pin_memory=True)
    host2 = torch.randint(-127, 128, (8, H, I), dtype=DT, pin_memory=True)
    h13s = torch.rand(8, W13_ROW, pin_memory=True)
    h2s = torch.rand(8, H, pin_memory=True)
    bytes_per_expert = host13[0].numel() + host2[0].numel() + (h13s[0].numel() + h2s[0].numel()) * 4

    lat = []
    for rep in range(30):
        e = rep % 8
        slot = rep % E
        t0 = time.perf_counter()
        w13[slot].copy_(host13[e], non_blocking=True)
        w2[slot].copy_(host2[e], non_blocking=True)
        scale13[slot].copy_(h13s[e], non_blocking=True)
        scale2[slot].copy_(h2s[e], non_blocking=True)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1e6)

    # batched variant: gather+scatter for several experts with one sync
    batch = []
    for rep in range(20):
        t0 = time.perf_counter()
        for e in range(4):
            slot = (rep * 4 + e) % E
            w13[slot].copy_(host13[e], non_blocking=True)
            w2[slot].copy_(host2[e], non_blocking=True)
        torch.cuda.synchronize()
        batch.append((time.perf_counter() - t0) * 1e6 / 4)

    lat.sort()
    print(f"expert slot payload = {bytes_per_expert/1e6:.2f} MB per (expert, rank)")
    print(f"single-expert fetch latency: median {statistics.median(lat):.0f} us, "
          f"p90 {lat[int(len(lat)*0.9)]:.0f} us, min {lat[0]:.0f} us")
    print(f"4-expert batched fetch per expert: median {statistics.median(batch):.0f} us")
    eff = bytes_per_expert / (statistics.median(lat) / 1e6) / 1e9
    print(f"effective bandwidth incl. launch+sync overhead: {eff:.1f} GB/s")
    print(f"=> 0.34 missed experts/layer (25% offload, 96.6% hit) = "
          f"{0.34*statistics.median(lat):.0f} us/layer vs 440 us layer budget")


if __name__ == "__main__":
    main()
