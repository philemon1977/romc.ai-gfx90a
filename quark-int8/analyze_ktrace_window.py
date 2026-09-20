#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 marker 切窗口聚合 rocprofv3 kernel-trace：decode 一步的时间到底归谁。

为什么需要它：rocprofv3 会把**整个进程生命周期**（含 5 分钟装载期）都抓下来，
不切窗口的话"GPU 忙碌占比"没有意义。prof_step.py 在计时窗口两端各放了一个
_mi250_prof_marker kernel，这里按它切。

用法：analyze_ktrace_window.py KTRACE_DIR [top_n]
"""
import csv
import glob
import os
import re
import sys
from collections import defaultdict

GROUPS = {
    "MoE(路由/专家/sort)": r"moe|expert|topk|sorting|grouped|align_block",
    "W4A16 反量化 GEMM": r"wna16|awq|dequant|int4|gptq|gemv",
    "attention/paged": r"attention|paged|flash|splitkv|reshape_and_cache|mla",
    "indexer(DCP/稀疏)": r"indexer|logits|ragged|dcp|merge|pack",
    "norm": r"norm",
    "allreduce/通信": r"allreduce|reduce_scatter|nccl|rccl|allgather",
    "elementwise/copy/cast": r"elementwise|copy|cast|concat|Cat|permute",
    "gemm(其它)": r"gemm|Cijk|matmul|linear",
}


def load(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                st = float(r.get("Start_Timestamp") or 0)
                en = float(r.get("End_Timestamp") or 0)
                du = float(r.get("Duration") or (en - st) or 0)
            except (TypeError, ValueError):
                continue
            if du <= 0:
                continue
            rows.append((st, en, du, r.get("Kernel_Name") or r.get("Name") or "?"))
    rows.sort()
    return rows


def window(rows):
    """按 marker 切出计时窗口；(rows, 说明)。"""
    mk = [r for r in rows if "prof_marker" in r[3]]
    if len(mk) >= 2:
        lo, hi = mk[0][0], mk[1][1]
        return [r for r in rows if r[0] >= lo and r[1] <= hi], "marker 窗口"
    return rows, "**无 marker，退化为整条 trace**"


def main():
    d = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    files = sorted(glob.glob(os.path.join(d, "**", "*kernel*trace*.csv"), recursive=True)) \
        or sorted(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True))
    if not files:
        print("no csv under", d, "->", os.listdir(d)[:20])
        return 1
    print("files:")
    for f in files:
        print("   %8.1f MiB  %s" % (os.path.getsize(f) / 2**20, f))
    allrows = []
    for f in files:
        allrows += load(f)
    allrows.sort()
    if not allrows:
        print("no parsable rows")
        return 1
    unit = 1e9 if max(r[1] for r in allrows) > 1e12 else 1e6
    rows, how = window(allrows)
    print("\n== 窗口：%s；kernels=%d（全 trace %d）==" % (how, len(rows), len(allrows)))
    lo = min(r[0] for r in rows)
    hi = max(r[1] for r in rows)
    wall = (hi - lo) / unit
    cov = 0.0
    cs = ce = None
    for st, en, du, _ in rows:
        if cs is None:
            cs, ce = st, en
        elif st <= ce:
            ce = max(ce, en)
        else:
            cov += ce - cs
            cs, ce = st, en
    cov = (cov + (ce - cs)) / unit
    busy = sum(r[2] for r in rows) / unit
    print("wall=%.1f ms  sum(dur)=%.1f ms  merged-coverage=%.1f ms ⇒ GPU 忙 %.1f%% / 空隙 %.1f%%"
          % (wall * 1e3, busy * 1e3, cov * 1e3, 100 * cov / wall, 100 * (1 - cov / wall)))
    agg = defaultdict(lambda: [0.0, 0])
    for _, _, du, nm in rows:
        a = agg[nm]
        a[0] += du / unit
        a[1] += 1
    print("\n%-64s %10s %7s %8s %9s" % ("kernel", "total(ms)", "share", "n", "avg(us)"))
    for nm, (t, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top_n]:
        print("%-64s %10.2f %6.1f%% %8d %9.1f" % (nm[:64], t * 1e3, 100 * t / busy, n, 1e6 * t / n))
    print("\n--- 按类目归并（占 busy 比例）---")
    for label, pat in GROUPS.items():
        rx = re.compile(pat, re.I)
        t = sum(du / unit for _, _, du, nm in rows if rx.search(nm))
        n = sum(1 for _, _, _, nm in rows if rx.search(nm))
        print("%-24s %9.2f ms %6.1f%%  %7d launches" % (label, t * 1e3, 100 * t / busy, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
