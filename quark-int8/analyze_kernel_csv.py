#!/usr/bin/env python3
"""分析 rocprofv3 kernel-trace CSV：把每个 kernel 的 **workgroup 数** 与 104 CU 对齐。

为什么是这个诊断（前序 docs/research-notes/aiter-cdna2-4-量化与GEMM.md §B5 点名的"唯一一个新高价值动作"）：
  8107 的 decode **离访存地板 12–16×**（地板只占 TPOT 的 6–8%）⇒ 瓶颈不在带宽，而在内核**占用/发射**。
  workgroup 数 ≪ 104（MI250X 每 GCD 的 CU 数）= 每步大半个 GPU 闲置，与算术无关。
用法：analyze_kernel_csv.py <csv 或目录> [--top 25] [--cu 104]
"""
import csv
import glob
import os
import sys
from collections import defaultdict

CU = 104


def pick(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += glob.glob(os.path.join(p, "**", "*kernel*trace*.csv"), recursive=True)
            out += glob.glob(os.path.join(p, "**", "*.csv"), recursive=True)
        else:
            out.append(p)
    # 去重、只留含 kernel 列的
    seen, keep = set(), []
    for f in out:
        if f in seen:
            continue
        seen.add(f)
        try:
            with open(f) as fh:
                head = fh.readline()
            if "Kernel_Name" in head or "Name" in head:
                keep.append(f)
        except Exception:
            pass
    return keep


def col(fields, *names):
    for n in names:
        for f in fields:
            if f.lower() == n.lower():
                return f
    return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    top = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--top=")), 25))
    cu = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--cu=")), CU))
    files = pick(args)
    if not files:
        raise SystemExit("没找到 kernel trace CSV")
    print(f"# 文件: {len(files)}   CU 基准 = {cu}")
    per = defaultdict(lambda: {"n": 0, "t": 0.0, "grids": defaultdict(int), "wg": defaultdict(int)})
    for f in files:
        with open(f, newline="") as fh:
            r = csv.DictReader(fh)
            fields = r.fieldnames or []
            ck = col(fields, "Kernel_Name", "Name")
            cd = col(fields, "Duration", "DurationNs", "Duration_Sum")
            cg = col(fields, "Grid_Size_X", "GridX")
            cgy = col(fields, "Grid_Size_Y", "GridY")
            cgz = col(fields, "Grid_Size_Z", "GridZ")
            cw = col(fields, "Workgroup_Size_X", "WorkgroupX")
            if not ck:
                continue
            for row in r:
                k = (row.get(ck) or "").strip()
                if not k:
                    continue
                try:
                    d = float(row.get(cd) or 0)
                except ValueError:
                    continue
                if d > 1e6:          # ns → ms 启发式
                    d /= 1e6
                elif d > 1e3:
                    d /= 1e3
                gx = int(float(row.get(cg) or 1)) if cg else 1
                gy = int(float(row.get(cgy) or 1)) if cgy else 1
                gz = int(float(row.get(cgz) or 1)) if cgz else 1
                w = int(float(row.get(cw) or 64)) if cw else 64
                e = per[k]
                e["n"] += 1
                e["t"] += d
                e["grids"][gx * gy * gz] += 1
                e["wg"][w] += 1
    total = sum(e["t"] for e in per.values()) or 1.0
    print(f"# kernel 种类 {len(per)}，总 GPU 时间 {total:.1f} ms")
    print(f"{'kernel':<58} {'n':>5} {'总ms':>9} {'占比':>6} {'中位grid':>9} {'占用(CU)':>9} {'wg':>5}")
    for k, e in sorted(per.items(), key=lambda x: -x[1]["t"])[:top]:
        grids = sorted(e["grids"])
        med = grids[len(grids) // 2]
        occ = min(100.0, 100.0 * med / cu)
        wgs = sorted(e["wg"])
        wg = wgs[len(wgs) // 2]
        print(f"{k[:58]:<58} {e['n']:>5} {e['t']:>9.1f} {100*e['t']/total:>5.1f}% "
              f"{med:>9} {occ:>8.0f}% {wg:>5}")
    idle = [k for k, e in per.items()
            if sorted(e["grids"])[len(e["grids"]) // 2] < cu and e["t"] / total > 0.01]
    print(f"\n# grid < {cu} 且占比 >1% 的 kernel：{len(idle)} 个（这些就是'每步大半个 GPU 闲置'的嫌疑）")
    for k in sorted(idle, key=lambda x: -per[x]["t"])[:10]:
        print(f"   {k[:70]}  grid中位={sorted(per[k]['grids'])[len(per[k]['grids'])//2]}  "
              f"占比={100*per[k]['t']/total:.1f}%")


if __name__ == "__main__":
    main()
