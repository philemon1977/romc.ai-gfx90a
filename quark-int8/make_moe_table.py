#!/usr/bin/env python3
"""Generate a vLLM Triton-MoE tuning table for MI250X.

Why: vLLM ships 331 tuning JSONs and NONE for device_name=AMD_Instinct_MI250X_MI250
(the server log literally says "Config file not found at .../E=512,N=128,
device_name=AMD_Instinct_MI250X_MI250,dtype=int4_w4a16.json"), so our MoE runs on
vLLM's default WNA16 heuristic: BM=16, BN=64, BK=32, GROUP_SIZE_M=1 (ROCm path,
num_valid_tokens//real_top_k != 1). davetha's precedent: hand-written MI250X MoE tile
tables gave +18% aggregate.

The lookup picks the key CLOSEST to the MoE call's token count M (single-stream with
MTP(5) => M=6), so every key 1..1024 is filled with the SAME candidate config to make
a clean single-variable A/B.

usage: make_moe_table.py OUT.json BM BN BK GROUP_M NUM_WARPS NUM_STAGES
"""
import json
import sys

out, bm, bn, bk, gm, warps, stages = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]),
)
entry = {
    "BLOCK_SIZE_M": bm,
    "BLOCK_SIZE_N": bn,
    "BLOCK_SIZE_K": bk,
    "GROUP_SIZE_M": gm,
    "SPLIT_K": 1,
    "num_warps": warps,
    "num_stages": stages,
}
keys = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
table = {str(k): dict(entry) for k in keys}
table["triton_version"] = "3.5.0"
with open(out, "w") as f:
    json.dump(table, f, indent=1)
print(f"wrote {out}  BM={bm} BN={bn} BK={bk} GROUP_M={gm} warps={warps} stages={stages}  keys={len(keys)}")
