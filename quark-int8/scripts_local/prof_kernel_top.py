#!/usr/bin/env python3
"""汇总 torch profiler chrome trace 里的 kernel 时间（按总时长排序）。"""
import glob, json, os, sys
from collections import defaultdict
root = sys.argv[1] if len(sys.argv) > 1 else "."
files = sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)) + sorted(glob.glob(os.path.join(root, "*.json.gz")))
if not files:
    print("(没有 trace 文件)", flush=True); sys.exit(0)
path = files[-1]
print("trace:", os.path.basename(path))
if path.endswith(".gz"):
    import gzip; data = json.load(gzip.open(path))
else:
    data = json.load(open(path))
ev = data.get("traceEvents", data if isinstance(data, list) else [])
agg = defaultdict(lambda: [0.0, 0])
for e in ev:
    if e.get("ph") == "X" and "cat" in e and e["cat"] in ("kernel", "gpu_memcpy", "gpu_memset"):
        name = e.get("name", "?").strip()[:64]
        agg[name][0] += float(e.get("dur", 0.0))
        agg[name][1] += 1
tot = sum(v[0] for v in agg.values()) or 1.0
rows = sorted(agg.items(), key=lambda kv: -kv[1][0])[:20]
print("  总 GPU 时间(us): %.1f  内核数: %d" % (tot, sum(v[1] for v in agg.values())))
for name, (dur, cnt) in rows:
    print("  %7.2f%%  %8.0f us  x%-5d  %s" % (100.0 * dur / tot, dur, cnt, name))
