#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最优 scale 的 A/B 量尺：用**生产函数** pack_int4_rows 本身，对比开/关 --optimal-scale。

## 为什么必须用生产函数来量

"最优 scale 能把误差从 10.0% 降到 6.8%"这个结论来自 `int4_scale_optimality.py`，
那是**另写的一版**搜索实现。若直接把那份数字当成转换器的预期收益，就可能重演
"两把尺子量同一件事"的错误。所以这里 import 转换器里的 `pack_int4_rows` 本身，
只切换 `_OPT_SCALE_FGRID`（生产代码里真正的开关），再逐张量比误差。

## 量尺

对源权重 w（fp4 专家 / fp8 注意力 / fp8 共享专家）：
    rec = 反量化(pack_int4_rows(w))          # 走完整打包→解包链路
    rel = ||rec - w|| / ||w||                # 范数比：全正项，无相消（上次 cos 伪影的教训）
    cos = F.cosine_similarity(rec, w)        # 稳定实现，不用手写 float32 点积
"""
import json
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, "/w/quark-int8")
import convert_dsv41_ct_int4 as C  # noqa: E402

SRC = "/src"


def unpack(packed, scale, group=32):
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    q = ((packed.unsqueeze(-1) >> sh) & 0xF).reshape(packed.shape[0], -1).to(torch.float32) - 8.0
    return q * scale.float().repeat_interleave(group, dim=1)


def run(w, grid):
    C._OPT_SCALE_FGRID = tuple(grid)
    pk, sc = C.pack_int4_rows(w)
    rec = unpack(pk, sc)
    ref = w.to(torch.float64).flatten()
    got = rec.to(torch.float64).flatten()
    rel = ((got - ref).norm() / ref.norm()).item()
    cos = F.cosine_similarity(rec.flatten(), w.float().flatten(), dim=0).item()
    return rel, cos


def main():
    grid = []
    f = 0.72
    while f <= 1.0 + 1e-9:
        grid.append(round(f, 4))
        f += 0.02
    print(f"栅格 {len(grid)} 点: {grid[0]}..{grid[-1]}", flush=True)

    wm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    cache = {}

    def srct(name):
        fsh = wm[name]
        if fsh not in cache:
            cache[fsh] = safe_open(f"{SRC}/{fsh}", framework="pt")
        return cache[fsh]

    cases = []
    for L in (0, 15, 39):
        for e in (0, 100):
            for w in ("w1", "w3"):
                cases.append(("fp4专家", L, f"layers.{L}.ffn.experts.{e}.{w}"))
        for w in ("w1", "w2"):
            cases.append(("fp8共享", L, f"layers.{L}.ffn.shared_experts.{w}"))
    for L in (0, 15, 39):
        for w in ("wkv", "wq_a"):
            cases.append(("fp8注意力", L, f"layers.{L}.attn.{w}"))

    tot_o = tot_n = 0.0
    cnt = 0
    print(f"{'类别':>10} {'层':>3} {'张量':>42} | {'原 rel':>9} {'原 cos':>9} | {'新 rel':>9} {'新 cos':>9} | {'改善':>7}")
    for kind, L, mod in cases:
        f_ = srct(f"{mod}.weight")
        s_ = srct(f"{mod}.scale")
        w = (C.dequant_fp4_expert(f_.get_tensor(f"{mod}.weight"), s_.get_tensor(f"{mod}.scale"))
             if kind == "fp4专家" else
             C.dequant_fp8_block(f_.get_tensor(f"{mod}.weight"), s_.get_tensor(f"{mod}.scale")))
        ro, co = run(w, ())
        rn, cn = run(w, grid)
        tot_o += ro
        tot_n += rn
        cnt += 1
        print(f"{kind:>10} {L:>3} {mod:>42} | {ro:>9.4%} {co:>9.5f} | {rn:>9.4%} {cn:>9.5f} | "
              f"{ro/rn:>6.3f}×", flush=True)
        del w
    print(f"\n=== 平均：原 {tot_o/cnt:.4%} → 新 {tot_n/cnt:.4%}  （改善 {tot_o/tot_n:.3f}×，n={cnt}）===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
