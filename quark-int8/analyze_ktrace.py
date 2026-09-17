#!/usr/bin/env python3
"""Aggregate a rocprofv3 kernel-trace CSV: what owns the decode step, and how much of
the wall time the GPU is actually busy (kernels) vs gap.

usage: analyze_ktrace.py KTRACE_DIR [top_n]
"""
import csv
import glob
import os
import re
import sys
from collections import defaultdict


def main():
    d = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    files = [f for f in glob.glob(os.path.join(d, "**", "*kernel*trace*.csv"), recursive=True)
             or glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)]
    if not files:
        print("no kernel-trace csv under", d)
        print("files present:", os.listdir(d)[:20])
        return 1
    path = max(files, key=os.path.getsize)
    print(f"trace: {path} ({os.path.getsize(path)/2**20:.1f} MiB)")

    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                st = float(r.get("Start_Timestamp") or 0)
                en = float(r.get("End_Timestamp") or 0)
                du = float(r.get("Duration") or (en - st) or 0)
            except (TypeError, ValueError):
                continue
            name = r.get("Kernel_Name") or r.get("Name") or "?"
            rows.append((st, en, du, name))
    if not rows:
        print("no parsable rows; header:", next(iter(csv.DictReader(open(path))), None))
        return 1

    rows.sort()
    unit = 1e9 if max(r[1] for r in rows) > 1e12 else 1e6  # ns or us
    lo = min(r[0] for r in rows)
    hi = max(r[1] for r in rows)
    wall = (hi - lo) / unit
    busy = sum(r[2] for r in rows) / unit
    # merged coverage (union of kernel intervals) = real busy wall time
    cov = 0.0
    cur_s = cur_e = None
    for st, en, du, _ in rows:
        if cur_s is None:
            cur_s, cur_e = st, en
        elif st <= cur_e:
            cur_e = max(cur_e, en)
        else:
            cov += cur_e - cur_s
            cur_s, cur_e = st, en
    cov = (cov + (cur_e - cur_s)) / unit

    print(f"kernels={len(rows)}  wall={wall*1e3:.1f} ms  sum(dur)={busy*1e3:.1f} ms  "
          f"merged-coverage={cov*1e3:.1f} ms")
    print(f"  → GPU busy fraction (merged) = {100*cov/wall:.1f}%   "
          f"gap/idle = {100*(1-cov/wall):.1f}%")

    agg = defaultdict(lambda: [0.0, 0])
    for st, en, du, name in rows:
        a = agg[name]
        a[0] += du / unit
        a[1] += 1
    print(f"\n{'kernel':<72} {'total(ms)':>10} {'share':>7} {'n':>7} {'avg(us)':>9}")
    for name, (t, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top_n]:
        print(f"{name[:72]:<72} {t*1e3:>10.2f} {100*t/busy:>6.1f}% {n:>7} {1e6*t/n:>9.1f}")

    groups = {
        "MoE (fused/expert/topk)": r"moe|expert|topk|sorting|grouped",
        "W4A16 dequant-gemm": r"wna16|awq|dequant|int4|gptq",
        "attention/paged": r"attention|paged|flash|splitkv|reshape_and_cache|cache_kernel",
        "norm": r"norm",
        "allreduce/comm": r"allreduce|AllReduce|reduce_scatter|nccl|rccl|reduce",
        "elementwise/copy/cast": r"elementwise|copy|cast|concat|index|Cat",
        "gemm(other)": r"gemm|Cijk|matmul",
    }
    print("\n--- 按类目归并 ---")
    for label, pat in groups.items():
        rx = re.compile(pat, re.I)
        t = sum(du / unit for _, _, du, nm in rows if rx.search(nm))
        n = sum(1 for _, _, _, nm in rows if rx.search(nm))
        print(f"{label:<24} {t*1e3:>9.2f} ms  {100*t/busy:>5.1f}% of busy  {n:>7} launches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
