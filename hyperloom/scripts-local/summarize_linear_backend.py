#!/usr/bin/env python3
"""汇总 linear_backend 扫描：bench 结果 + server 日志的稳态采样。

bench 的 output_throughput 含启动爬坡（首次完成时间 auto 2:27 / aiter 2:29 / triton 3:10），
所以另外从 server 日志取"满载采样"（Running >= 32）的中位数做同口径对比。
"""
import json
import os
import re
import statistics
import sys

P = sys.argv[1]
hdr = "%-8s %10s %14s %9s" % ("backend", "bench_tput", "steady_median", "steady_n")
print(hdr)
print("-" * len(hdr))

for lb in ("auto", "aiter", "triton", "torch"):
    vals = []
    f = os.path.join(P, "server-%s.log" % lb)
    if os.path.exists(f):
        for line in open(f, errors="replace"):
            m = re.search(r"Avg generation throughput: ([0-9.]+) tokens/s.*Running: (\d+) reqs", line)
            if m and int(m.group(2)) >= 32:
                vals.append(float(m.group(1)))
    tput = ""
    j = os.path.join(P, "lb-%s.json" % lb)
    if os.path.exists(j):
        tput = round(json.load(open(j))["output_throughput"], 2)
    med = round(statistics.median(vals), 1) if vals else None
    print("%-8s %10s %14s %9d" % (lb, tput, med, len(vals)))
