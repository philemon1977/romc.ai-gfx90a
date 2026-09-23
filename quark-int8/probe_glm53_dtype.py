# -*- coding: utf-8 -*-
"""纯标准库解析 safetensors 头：统计 dtype 分布、参数量、字节数（决定 int8/int4 能否装进 512GB）。"""
import json, os, struct, collections, sys

M = sys.argv[1]
idx = json.load(open(os.path.join(M, "model.safetensors.index.json")))["weight_map"]
files = sorted(set(idx.values()))
print("分片数:", len(files), " 张量数:", len(idx))

by_dtype = collections.Counter()      # dtype -> 字节
params = collections.Counter()        # dtype -> 元素数
nscale = 0
DT = {"F8_E4M3": 1, "F8_E5M2": 1, "BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1, "I32": 4, "I64": 8, "BOOL": 1}
big = []
for fn in files[:6] + files[-2:]:
    with open(os.path.join(M, fn), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        dt = v["dtype"]; nb = v["data_offsets"][1] - v["data_offsets"][0]
        by_dtype[dt] += nb
        el = DT.get(dt, 1)
        params[dt] += nb // el
        if "scale" in k or "scale_inv" in k:
            nscale += 1
        if nb > 200 * 2**20:
            big.append((nb / 2**20, k, dt, tuple(v["shape"])))
sample_files = len(files[:6] + files[-2:])
tot_bytes = sum(by_dtype.values())
print("抽样 %d/%d 分片:" % (sample_files, len(files)))
for dt, nb in by_dtype.most_common():
    print("   %-8s %10.2f GiB   元素 %13d" % (dt, nb / 2**30, params[dt]))
print("   抽样合计 %.2f GiB   -> 全模型估算 %.1f GiB" % (tot_bytes / 2**30, tot_bytes / 2**30 / sample_files * len(files)))
print("scale 类张量数(抽样):", nscale)
print("大张量示例:")
for nb, k, dt, sh in sorted(big, reverse=True)[:6]:
    print("   %8.1f MiB %-52s %-8s %s" % (nb, k, dt, sh))
