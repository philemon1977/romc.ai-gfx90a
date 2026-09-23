#!/usr/bin/env python3
"""engram 运行时探针的离线参考：从 checkpoint 分片里解码指定 id 的行。

用法:
  python3 engram_row_ref.py <shard> <id> [<id> ...]
例（layer 1 的 engram 在 shard 47）:
  python3 engram_row_ref.py model-00047-of-00048.safetensors 123456 78901

输出与内核探针 [ENG-DBG] 的字段一一对应：
  (absolute_id, scale_byte, [code0..code7])   # 前 4 字节 = 列 0..7，低 nibble 在前、两补码
"""
import json
import os
import struct
import sys

D = "/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4"


def main():
    shard = sys.argv[1]
    ids = [int(x) for x in sys.argv[2:]]
    path = os.path.join(D, shard)
    with open(path, "rb") as f:
        hn = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(hn))
        base = 8 + hn
        wname = [k for k in h if k.endswith(".engram.embed.weight")][0]
        sname = wname[: -len("weight")] + "scale"
        w_off, w_shape = h[wname]["data_offsets"][0], h[wname]["shape"]
        s_off, s_shape = h[sname]["data_offsets"][0], h[sname]["shape"]
        row_bytes, s_cols = w_shape[1], s_shape[1]
        out = []
        for i in ids:
            f.seek(base + w_off + i * row_bytes)
            packed = f.read(4)
            f.seek(base + s_off + i * s_cols)
            scale = f.read(1)[0]
            codes = []
            for b in packed:
                lo, hi = b & 0xF, (b >> 4) & 0xF
                codes += [lo - 16 if lo >= 8 else lo, hi - 16 if hi >= 8 else hi]
            out.append((i, scale, codes))
    print(f"{os.path.basename(path)}  rows={w_shape[0]} row_bytes={row_bytes} scale_cols={s_cols}")
    for i, sc, codes in out:
        print(f"  ({i}, {sc}, {codes})")


if __name__ == "__main__":
    main()
