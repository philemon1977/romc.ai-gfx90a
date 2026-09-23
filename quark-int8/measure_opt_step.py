#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""改 `--opt-scale-step` 默认值之前的量尺：6 点栅格相对 15 点，**损失多少保真度、省多少时间**。

纪律依据：不能凭"SSE 曲线在极值附近很平"这种口头推理就改默认值，必须用**生产函数**
（`convert_dsv41_ct_int4.pack_int4_rows`，即真正写盘的那段代码）量出：
    rel(栅格)  = ||dequant(pack(w)) - w|| / ||w||     （范数比：全正项，无相消 —— 上次的教训）
    time(栅格) = 同批张量重复 K 次的墙钟
本脚本刻意**限线程 + nice**，避免干扰正在跑的转换（它现在是 CPU 瓶颈）。
"""
import json
import sys
import time

import torch
from safetensors import safe_open

sys.path.insert(0, "/w/quark-int8")
import convert_dsv41_ct_int4 as C  # noqa: E402

SRC = "/src"
REPS = 6  # 重复次数，压掉单次抖动


def grid(step, lo=0.72, hi=1.00):
    g, f = [], lo
    while f <= hi + 1e-9:
        g.append(round(f, 4))
        f += step
    return tuple(g)


def main():
    wm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    names = ["layers.0.ffn.experts.0.w1", "layers.0.ffn.experts.100.w3",
             "layers.0.ffn.shared_experts.w1", "layers.0.attn.wq_b"]
    tensors, handles = [], {}
    for n in names:
        f = wm[n + ".weight"]
        if f not in handles:
            handles[f] = safe_open(f"{SRC}/{f}", framework="pt")
        h = handles[f]
        w = h.get_tensor(n + ".weight")
        s = h.get_tensor(n + ".scale")
        # ★ 必须按**源格式**分别反量化：最优栅格收益依赖值分布，
        #   而 fp4 专家（重尾）与 fp8 block（宽动态范围）的改善率差别很大
        #   （实测 1.48–1.61× vs 1.12–1.14×），统一解会让结论失真。
        if "ffn.experts." in n:
            w0 = C.dequant_fp4_expert(w, s, dtype=torch.float32)
            kind = "fp4"
        else:
            w0 = C.dequant_fp8_block(w, s, dtype=torch.float32)
            kind = "fp8"
        tensors.append((n, w0, kind))
    print(f"样本 {len(tensors)} 个张量，重复 {REPS} 次；线程上限 {torch.get_num_threads()}", flush=True)

    base_rel = base_t = None
    for step in (0.02, 0.05, 0.10, 0.50):
        C._OPT_SCALE_FGRID = grid(step)
        npts = len(C._OPT_SCALE_FGRID)
        rels = []
        for n, w, kind in tensors:
            pk, sc = C.pack_int4_rows(w)
            shf = torch.arange(0, 32, 4, dtype=torch.int32)
            q = ((pk.unsqueeze(-1) >> shf) & 0xF).reshape(pk.shape[0], -1).float() - 8.0
            rec = q * sc.float().repeat_interleave(32, dim=1)
            a, b = rec.double().flatten(), w.double().flatten()
            rels.append(((a - b).norm() / b.norm()).item())
        mean_rel = sum(rels) / len(rels)
        by_kind = {}
        for (n, w, kind), r in zip(tensors, rels):
            by_kind.setdefault(kind, []).append(r)
        # 计时：重复 REPS 次同样的 4 个张量，只比相对值，不比绝对值
        t0 = time.time()
        for _ in range(REPS):
            C._OPT_SCALE_FGRID = grid(step)
            for n, w, kind in tensors:
                C.pack_int4_rows(w)
        dt = (time.time() - t0) / REPS
        if base_rel is None:
            base_rel, base_t = mean_rel, dt
            ks = "  ".join(f"{k}:{sum(v)/len(v):.4%}" for k, v in sorted(by_kind.items()))
            print(f"  step={step:<5} 点数={npts:2d}  rel={mean_rel:.4%}  每次={dt:5.2f}s   （基准） {ks}",
                  flush=True)
        else:
            ks = "  ".join(f"{k}:{sum(v)/len(v):.4%}" for k, v in sorted(by_kind.items()))
            print(f"  step={step:<5} 点数={npts:2d}  rel={mean_rel:.4%}  每次={dt:5.2f}s  "
                  f"→ rel 变差 {(mean_rel/base_rel-1)*100:+.3f}%  耗时 {dt/base_t:.2f}×  {ks}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
