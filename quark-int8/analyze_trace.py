#!/usr/bin/env python3
"""Aggregate a vLLM torch-profiler trace: which kernels own the decode step, and how
many launches happen per step. This gives the Amdahl ceiling for any custom kernel.

Usage: analyze_trace.py PROF_DIR [top_n]
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict


def load_events(path):
    with open(path) as f:
        d = json.load(f)
    ev = d.get("traceEvents", d if isinstance(d, list) else [])
    return ev


def main():
    prof_dir = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    cands = []
    for pat in ("**/*.pt.trace.json*", "**/*.json", "**/*.json.gz"):
        cands += glob.glob(os.path.join(prof_dir, pat), recursive=True)
    cands = sorted(set(c for c in cands if os.path.getsize(c) > 100_000),
                   key=os.path.getsize, reverse=True)
    if not cands:
        print("no trace files found under", prof_dir)
        return 1
    path = cands[0]
    print(f"trace: {path}  ({os.path.getsize(path)/2**20:.1f} MiB)")
    if path.endswith(".gz"):
        import gzip
        with gzip.open(path) as f:
            d = json.load(f)
        ev = d.get("traceEvents", [])
    else:
        ev = load_events(path)
    print(f"events: {len(ev)}")

    kernels = [e for e in ev if e.get("cat") == "kernel" and e.get("dur")]
    if not kernels:
        print("no kernel events (cat=='kernel')")
        return 1

    tot = sum(e["dur"] for e in kernels)
    span_lo = min(e["ts"] for e in kernels)
    span_hi = max(e["ts"] + e["dur"] for e in kernels)
    span = span_hi - span_lo
    print(f"kernel total = {tot/1e6:.1f} s over wall span {span/1e6:.1f} s "
          f"({100*tot/span:.1f}% busy) | {len(kernels)} launches")

    agg = defaultdict(lambda: [0.0, 0])      # name -> [us, count]
    for e in kernels:
        a = agg[e["name"]]
        a[0] += e["dur"]
        a[1] += 1
    rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
    print(f"\n{'kernel':<78} {'total(s)':>9} {'share':>7} {'n':>7} {'avg(us)':>9}")
    for name, (us, n) in rows[:top_n]:
        print(f"{name[:78]:<78} {us/1e6:>9.3f} {100*us/tot:>6.1f}% {n:>7} {us/n:>9.1f}")

    groups = {
        "MoE (fused/expert)": r"moe|expert|topk|sorting|m_grouped|grouped_gemm",
        "W4A16/int4 dequant-gemm": r"wna16|awq|dequant|int4|gptq|marlin",
        "attention/paged": r"attention|paged|flash|splitkv|cache_kernel|reshape_and_cache",
        "rmsnorm/layernorm": r"norm",
        "elementwise/copy/cast": r"elementwise|copy|cast|CatArray|concat|index",
        "allreduce/comm": r"all_reduce|AllReduce|reduce_scatter|nccl|rccl|custom_reduce",
        "gemm (other)": r"gemm|Cijk|matmul",
    }
    print("\n--- 按类目归并（同一 kernel 可命中多类，仅作口径参考）---")
    for label, pat in groups.items():
        rx = re.compile(pat, re.I)
        us = sum(e["dur"] for e in kernels if rx.search(e["name"]))
        n = sum(1 for e in kernels if rx.search(e["name"]))
        print(f"{label:<26} {us/1e6:>8.3f} s  {100*us/tot:>5.1f}%  {n:>7} launches")

    # decode step count: use the vLLM step markers if present
    steps = [e for e in ev if "gpu_user_annotation" == e.get("cat")
             and re.search(r"execute_context|model_runner|draft|target", e.get("name", ""), re.I)]
    if steps:
        names = defaultdict(int)
        for e in steps:
            names[e["name"]] += 1
        print("\n--- step 标注计数（前 8）---")
        for k, v in sorted(names.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  {v:>6}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
