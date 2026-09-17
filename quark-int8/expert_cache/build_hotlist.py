#!/usr/bin/env python3
"""Build the per-layer expert hot-list from the routing data collected by
collect_routing.py (task ①).

Output: hotlist.npz with
  freq   : [layers, 512] int64   routing frequency per (layer, expert)
  order  : [layers, 512] int32   experts sorted by descending frequency
Used by the expert-cache prototype to decide which experts are "hot" (pinned in
HBM) and which are "cold" (offloaded to DRAM).
"""
import os
import sys

import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "/work/routing_samples.npz"
DST = sys.argv[2] if len(sys.argv) > 2 else "/work/hotlist.npz"

z = np.load(SRC)
arrs = [z[k] for k in z.files]
allr = np.concatenate([a.reshape(-1, a.shape[-2], a.shape[-1]) for a in arrs], axis=0)
T, L, K = allr.shape
freq = np.zeros((L, 512), dtype=np.int64)
for l in range(L):
    ids, cnt = np.unique(allr[:, l, :], return_counts=True)
    freq[l, ids] = cnt
order = np.argsort(-freq, axis=1).astype(np.int32)
np.savez_compressed(DST, freq=freq, order=order)
print(f"[hotlist] {T} tokens x {L} layers x top{K} -> {DST}")

tot = freq.sum(axis=1)
for pin in (320, 384, 416, 448, 480):
    hit = np.mean([freq[l][order[l][:pin]].sum() / tot[l] for l in range(L)])
    print(f"  pin {pin:3d}/512 ({100*(512-pin)/512:4.0f}% offloaded) -> hit {hit*100:5.1f}%")
