#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定向权重转换保真度：源模型(fp4/fp8) vs 我们的 CT-Int4，聚焦"异常层"。

## 动机（来自 v64 的 40 层健康画像）

逐层画像显示：残差流 rms 从 layer0 的 0.098 单调涨到 layer39 的 6.85（70×），
其中两处 FFN 输出尖峰异常突出：
    layer 15: rms_ffn_out = 1.417   （邻层 14/16 仅 0.205 / 0.159，差 ~7×）
    layer 39: rms_ffn_out = 5.483   （邻层 38 为 0.632，差 ~9×）

这两个数值**已由 mHC mixes（4e-7）与 hc_post 的用法（A/B 链）证明按设计走**，
所以剩下的解释只有两类：
    (a) 模型本来就该这样（需要参考实现才能判）；
    (b) **这两个层的权重在 int4 转换里被改坏了**。
(b) 可以**离线、无需 GPU** 地排掉：拿源模型的 fp4/fp8 权重与我们的 int4 g32 逐张量比 cos。
注意 `verify_ct_int4.py` 的全量校验是**每分片采样 64 张量**，未必命中这几层，
所以这里做的是**定向补测**（对照层 0 一起测）。

## 两边的反量化口径（各自独立、都在别处验证过）
  源 fp4 专家 : q=(hi,lo) 高nibble在前 → FP4_LUT[|q|] 带符号 → × e8m0(scale) 沿 K 重复 32
  源 fp8 共享 : w * e8m0(scale)，block 重复
  我们的 int4 : q=((packed>>[0,4,..,28])&0xF)-8（低nibble在前） → × scale 沿 K 重复 32
"""
import json
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, "/w/quark-int8")
from verify_ct_int4 import dequant_fp4_expert, dequant_fp8_block  # noqa: E402

SRC = "/src"
OUT = "/models"
GROUP = 32


def ours_int4(sh, prefix):
    packed = sh.get(f"{prefix}.weight_packed").to(torch.int32)
    shf = torch.arange(0, 32, 4, dtype=torch.int32)
    q = ((packed.unsqueeze(-1) >> shf) & 0xF).reshape(packed.shape[0], -1).to(torch.float32) - 8.0
    scale = sh.get(f"{prefix}.weight_scale").to(torch.float32)
    w = q * scale.repeat_interleave(GROUP, dim=1)
    shape = sh.get(f"{prefix}.weight_shape").tolist()
    return w.reshape(shape)


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


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-30)).item()


def main():
    src, ours = S(SRC), S(OUT)
    layers = [int(x) for x in (sys.argv[1:] or ["0", "15", "39"])]
    experts = [0, 7, 100, 383]
    print(f"{'层':>4} {'专家':>5} {'矩阵':>4} {'源-我们 cos':>12} {'相对误差':>10}")
    for L in layers:
        for e in experts:
            for w in ("w1", "w2", "w3"):
                sp = f"layers.{L}.ffn.experts.{e}.{w}"
                ref = dequant_fp4_expert(src.get(f"{sp}.weight"), src.get(f"{sp}.scale"))
                got = ours_int4(ours, sp)
                if list(ref.shape) != list(got.shape):
                    print(f"{L:>4} {e:>5} {w:>4}   形状不一致 {tuple(ref.shape)} vs {tuple(got.shape)}")
                    continue
                c = cos(ref, got)
                r = ((got - ref).norm() / (ref.norm() + 1e-30)).item()
                flag = "" if c > 0.99 else "  ◆差◆"
                print(f"{L:>4} {e:>5} {w:>4} {c:>12.6f} {r:>10.4%}{flag}")
        # 共享专家（源为 fp8 block）
        for w in ("w1", "w2", "w3"):
            sp = f"layers.{L}.ffn.shared_experts.{w}"
            ref = dequant_fp8_block(src.get(f"{sp}.weight"), src.get(f"{sp}.scale"))
            got = ours_int4(ours, sp)
            if list(ref.shape) != list(got.shape):
                print(f"{L:>4} {'共享':>5} {w:>4}   形状不一致 {tuple(ref.shape)} vs {tuple(got.shape)}")
                continue
            c = cos(ref, got)
            print(f"{L:>4} {'共享':>5} {w:>4} {c:>12.6f} {((got-ref).norm()/(ref.norm()+1e-30)).item():>10.4%}"
                  + ("" if c > 0.99 else "  ◆差◆"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
