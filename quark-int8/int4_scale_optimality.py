#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""int4 转换质量：我们存的 scale 是不是**最优**的？（决定"10% 误差"是固有极限还是转换缺陷）

## 为什么这是关键问题

v64 的定向比对发现：源模型(fp4 e2m1) → 我们的 int4 g32，cos 仅 0.96~0.978、相对误差 ~10%，
**且对所有层一致（含已被逐部件验证过的 layer 0）**。而 cos 0.962 配 10.2% 相对误差，
按「误差与真值正交」应有 cos = 1/sqrt(1+0.102²) = 0.9949 —— 实测 0.962 低得多，
⇒ 误差是**系统性收缩**（got ≈ 0.96·ref），不是随机噪声。

这不奇怪：源 fp4 e2m1 的栅格是 {0,0.5,1,1.5,2,3,4,6}（近零处密，适合权重分布），
而我们转成的是**均匀栅格**（int4 + 单一 scale，零点在 nibble=8）。
**fp4 → int4-uniform 本身就会掉精度**；问题只在于：我们**有没有把这个均匀栅格调到最优**。

## 测法

对每一组 32 个反量化后的源值 v：
  err_ours = || (nibble-8)*s_ours  -  v ||            （我们实际存的 scale）
  err_opt  = min_s || quant_s(v)  -  v ||             （对该组做 1 维 scale 搜索）
若 err_ours ≈ err_opt ⇒ 转换已到 int4 的极限（10% 是**固有**代价，换 GPTQ 也救不了多少）；
若 err_ours ≫ err_opt ⇒ **转换的 scale 选错了**，是**可修的缺陷**。
"""
import json
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, "/w/quark-int8")
from verify_ct_int4 import dequant_fp4_expert  # noqa: E402

SRC = "/src"
OUT = "/models"
GROUP = 32


class S:
    def __init__(self, d):
        with open(f"{d}/model.safetensors.index.json") as fh:
            self.map = json.load(fh)["weight_map"]
        self.d = d
        self.h = {}

    def _h(self, f):
        if f not in self.h:
            self.h[f] = safe_open(f"{self.d}/{f}", framework="pt")
        return self.h[f]

    def get(self, n):
        return self._h(self.map[n]).get_tensor(n)


def quant_s(v, s):
    """uint4b8：nibble = clamp(round(v/s)+8, 0, 15)，返回反量化值。"""
    n = torch.clamp(torch.round(v / s) + 8, 0, 15)
    return (n - 8) * s


def relerr(a, b):
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def main():
    layers = [int(x) for x in (sys.argv[1:] or ["0", "15", "39"])]
    src, ours = S(SRC), S(OUT)
    print(f"{'层':>4} {'专家':>5} {'矩阵':>4} | {'我们的 rel':>10} {'最优 rel':>10} {'劣化倍数':>9} | "
          f"{'s_ours/s_opt 中位':>17} {'源 absmax/7 对比':>16}")
    for L in layers:
        for e in (0, 100):
            for w in ("w1", "w3"):
                sp = f"layers.{L}.ffn.experts.{e}.{w}"
                ref = dequant_fp4_expert(src.get(f"{sp}.weight"), src.get(f"{sp}.scale")).float()
                packed = ours.get(f"{sp}.weight_packed").to(torch.int32)
                s_ours = ours.get(f"{sp}.weight_scale").float()
                sh = torch.arange(0, 32, 4, dtype=torch.int32)
                q = ((packed.unsqueeze(-1) >> sh) & 0xF).reshape(packed.shape[0], -1).to(torch.float32) - 8.0
                got = q * s_ours.repeat_interleave(GROUP, dim=1)

                N, K = ref.shape
                ng = K // GROUP
                v = ref.view(N, ng, GROUP)
                s_o = s_ours.view(N, ng, 1)
                e_ours = relerr((q.view(N, ng, GROUP) * s_o), v)

                # 1 维 scale 搜索（每组的 scale 独立）
                best = None
                ratios = []
                for f in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5]:
                    s_c = s_o * f
                    cand = quant_s(v, s_c)
                    err = (cand - v).pow(2).sum().item()
                    if best is None or err < best[0]:
                        best = (err, f)
                # 用最优 f 重新算相对误差
                e_opt = relerr(quant_s(v, s_o * best[1]), v)
                ratios.append(best[1])
                absmax = v.abs().amax(dim=-1, keepdim=True)
                print(f"{L:>4} {e:>5} {w:>4} | {e_ours:>10.4%} {e_opt:>10.4%} {e_ours/(e_opt+1e-30):>9.3f} | "
                      f"{best[1]:>17.2f} {(s_o/(absmax/7.0+1e-30)).median().item():>16.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
