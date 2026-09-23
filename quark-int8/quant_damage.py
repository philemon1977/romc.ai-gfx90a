#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把"权重误差"翻译成"功能损伤"：同一输入下，三种权重的 MoE 输出差多少？

## 为什么这个指标比 relRMSE 更有说服力

`validate_opt_scale.py` 量的是**权重**层面的 rel（10.0% → 6.8%）。但决定模型行为的是
**输出**层面的偏差。本脚本用 vLLM 真实请求 dump 下来的 `ffn_in`（layer0/2，T=22），
在同一套 gate（三者完全相同）下分别用三种权重跑 MoE，直接比输出：

    (a) 源权重（fp4 专家 + fp8 共享专家 反量化）= 理想基准
    (b) 旧 int4（现役 397 GiB 模型）
    (c) 新 int4（--optimal-scale 产物）

指标：相对偏差 ||y - y_a|| / ||y_a|| 与 cos。若 (c) 明显比 (b) 更接近 (a)，
说明"最优 scale"在**功能**上确实减轻了损伤，而不只是账面数字好看。

注意 gate 用的是源权重（三边一致），所以路由完全相同，差异只来自专家/共享专家的权重。
"""
import argparse
import json
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, "/w/quark-int8")
from convert_dsv41_ct_int4 import dequant_fp4_expert, dequant_fp8_block  # noqa: E402

G = 32
TOPK = 6


class Dir:
    def __init__(self, d):
        with open(f"{d}/model.safetensors.index.json") as fh:
            self.map = json.load(fh)["weight_map"]
        self.d = d
        self.h = {}
        self.is_source = "inference" not in d and self._looks_like_source()

    def _looks_like_source(self):
        return any(k.endswith("ffn.experts.0.w1.weight") for k in self.map)

    def _f(self, shard):
        if shard not in self.h:
            self.h[shard] = safe_open(f"{self.d}/{shard}", framework="pt")
        return self.h[shard]

    def get(self, name):
        return self._f(self.map[name]).get_tensor(name)

    def int4_mod(self, mod):
        pk = self.get(f"{mod}.weight_packed").to(torch.int32)
        sh = torch.arange(0, 32, 4, dtype=torch.int32)
        q = ((pk.unsqueeze(-1) >> sh) & 0xF).reshape(pk.shape[0], -1).to(torch.float32) - 8.0
        sc = self.get(f"{mod}.weight_scale").to(torch.float32)
        return (q * sc.repeat_interleave(G, dim=1)).reshape(self.get(f"{mod}.weight_shape").tolist())

    def expert(self, mod):
        if self.is_source:
            return dequant_fp4_expert(self.get(f"{mod}.weight"), self.get(f"{mod}.scale"),
                                      dtype=torch.float32)
        return self.int4_mod(mod)

    def shared(self, mod):
        if self.is_source:
            return dequant_fp8_block(self.get(f"{mod}.weight"), self.get(f"{mod}.scale"),
                                     dtype=torch.float32)
        # 新策略下共享专家是 bf16 未量化（没有 weight_packed）——这本身就是被测的改动
        if f"{mod}.weight_packed" not in self.map:
            return self.get(f"{mod}.weight").to(torch.float32)
        return self.int4_mod(mod)


def swiglu(x, w1, w3, w2, lim=10.0, wt=None):
    gate = x @ w1.t()
    up = x @ w3.t()
    up = torch.clamp(up, -lim, lim)
    gate = torch.clamp(gate, max=lim)
    h = F.silu(gate) * up
    if wt is not None:
        h = wt * h
    return h @ w2.t()


def moe(x, gate_w, gate_b, D, L):
    scores = F.softplus(x @ gate_w.t()).sqrt()
    idx = (scores + gate_b).topk(TOPK, dim=-1)[1]
    w = scores.gather(1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * 1.5
    y = torch.zeros_like(x)
    yr = torch.zeros_like(x)
    for e in idx.flatten().unique().tolist():
        pos, top = torch.where(idx == e)
        xe = x[pos]
        yr[pos] += swiglu(xe,
                          D.expert(f"layers.{L}.ffn.experts.{e}.w1"),
                          D.expert(f"layers.{L}.ffn.experts.{e}.w3"),
                          D.expert(f"layers.{L}.ffn.experts.{e}.w2"),
                          wt=w[pos, top, None])
    y += yr
    ys = swiglu(x,
                D.shared(f"layers.{L}.ffn.shared_experts.w1"),
                D.shared(f"layers.{L}.ffn.shared_experts.w3"),
                D.shared(f"layers.{L}.ffn.shared_experts.w2"))
    y += ys
    return y, yr, ys


def dev(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item(), F.cosine_similarity(
        a.flatten(), b.flatten(), dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/dump/dsv41_L0_moe.pt")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--src", default="/src")
    ap.add_argument("--old", default="/models")
    ap.add_argument("--new", default="/new")
    a = ap.parse_args()

    d = torch.load(a.dump, map_location="cpu")
    x = d["ffn_in"].float().reshape(-1, 5120)
    print(f"=== layer {a.layer}: ffn_in {tuple(x.shape)} rms={x.pow(2).mean().sqrt():.4f} ===", flush=True)

    S, O, N = Dir(a.src), Dir(a.old), Dir(a.new)
    gw = S.get(f"layers.{a.layer}.ffn.gate.weight").float()
    gb = S.get(f"layers.{a.layer}.ffn.gate.bias").float()

    res = {}
    res_r = {}
    res_s = {}
    for tag, D in (("源(理想)", S), ("旧int4", O), ("新int4", N)):
        y, yr, ys = moe(x, gw, gb, D, a.layer)
        res[tag] = y
        res_r[tag] = yr
        res_s[tag] = ys
        # ★ 关键分解：routed 与 shared 各自的量级。若某项在**源权重**下也远小于另一项，
        #   说明该比例是模型设计；若只有 int4 下才失衡，则是转换缺陷。
        print(f"  {tag:9s} rms: routed={yr.pow(2).mean().sqrt():.5f} "
              f"shared={ys.pow(2).mean().sqrt():.5f} full={y.pow(2).mean().sqrt():.5f}", flush=True)
        if tag == "源(理想)":
            print(f"  {tag:9s} （基准）", flush=True)
        else:
            r, c = dev(y, res["源(理想)"])
            rr, rc = dev(yr, res_r["源(理想)"])
            sr, sc_ = dev(ys, res_s["源(理想)"])
            print(f"  {tag:9s} vs 源: 全={r:.4%}  routed={rr:.4%} shared={sr:.4%}  cos={c:.6f}",
                  flush=True)

    r_old, _ = dev(res["旧int4"], res["源(理想)"])
    r_new, _ = dev(res["新int4"], res["源(理想)"])
    print(f"\n=== 功能损伤：旧 {r_old:.4%} → 新 {r_new:.4%}  （减轻 {r_old/max(r_new,1e-12):.3f}×）===",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
