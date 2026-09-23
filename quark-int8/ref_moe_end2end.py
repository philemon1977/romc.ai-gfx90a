#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MoE(FFN) 端到端对拍：用 checkpoint 权威语义复算 layer0 的 MoE 输出，与 vLLM dump 逐项对比。

## 为什么要有这个脚本

到 v62 为止，我们对 MoE 的验证是两把**局部**尺子：
  1) 「MoE 的 GEMM 内核对」——int4 GEMV 内核 vs torch 参考，误差 0.219%、cos 0.999998；
  2) 「路由选择对」——top-6 专家 id 与参考逐个相同。
但**从未验过 MoE 的最终输出**，即：加权 → 专家输出累加 → shared expert 相加。
而「知识进入残差流」的唯一入口正是这一步。若 shared expert 没被加上、或加权/累加有误，
模型就会退化成「只会复读上下文、不会用知识」——与观测症状（复读、确定性、解码≡预取）
完全吻合，且能同时解释「所有可单独测的部件都对，整体却退化」这一矛盾。

## 参考语义（逐行照抄 checkpoint 的 inference/model.py:792-904，不凭记忆）

  Gate.forward:                       # gate_temp=1.0, score_func=sqrtsoftplus, topk=6
      scores = linear(x.float(), W.float()) / gate_temp
      scores = F.softplus(scores).sqrt()
      idx    = (scores + bias).topk(topk, -1)[1]      # bias 只选专家、不参与缩放
      w      = scores.gather(1, idx)                  # 权重取自**无偏** scores
      w     /= w.sum(-1, keepdim=True) + 1e-20        # norm_topk_prob=True
      w     *= route_scale                            # 1.5
  Expert.forward:                     # swiglu_limit=10.0
      gate = w1(x).float(); up = w3(x).float()
      up   = clamp(up, -10, 10); gate = clamp(gate, max=10)
      h    = silu(gate) * up
      h    = weights * h                              # 权重乘在**激活**上，再进 w2
      return w2(h.to(x.dtype))
  MoE.forward:
      y = zeros_like(x, dtype=float32)
      逐专家: y[idx] += expert(x[idx], w[idx, top])
      y += self.shared_experts(x)                     # ★ 不加权、直接相加
      return y.type_as(x)

## int4 反量化（CT pack-quantized uint4b8，口径来自已验证的 verify_ct_int4.py）
  q = ((packed >> [0,4,...,28]) & 0xF) - 8      # 低 nibble 在前，offset 8
  w = q * scale.repeat_interleave(32, dim=1)    # int4 g32
"""
import argparse
import json
import struct
import sys

import torch
from safetensors import safe_open

DIM = 5120
INTER = 2304
N_EXPERTS = 384
TOPK = 6
GROUP = 32
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0


def unpack_int4(packed):
    """packed: int32 [N, K/8] -> float32 [N, K]（低 nibble 在前，offset 8）。"""
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    q = ((packed.unsqueeze(-1) >> sh) & 0xF).reshape(packed.shape[0], -1)
    return q.to(torch.float32) - 8.0


class Shards:
    """按张量名索引 safetensors 分片，只为需要的张量打开文件句柄。"""

    def __init__(self, model_dir):
        with open(f"{model_dir}/model.safetensors.index.json") as fh:
            self.map = json.load(fh)["weight_map"]
        self.dir = model_dir
        self._open = {}

    def _h(self, fname):
        if fname not in self._open:
            self._open[fname] = safe_open(f"{self.dir}/{fname}", framework="pt")
        return self._open[fname]

    def get(self, name):
        return self._h(self.map[name]).get_tensor(name)

    def dequant(self, prefix):
        """prefix 形如 layers.0.ffn.experts.12.w1 -> 反量化后的 float32 权重 [N,K]。"""
        packed = self.get(f"{prefix}.weight_packed")
        # 注意：safetensors 里 I32 是 int32；nibble 顺序低字节在前（已在 verify_ct_int4.py 验证）
        q = unpack_int4(packed.to(torch.int32))
        scale = self.get(f"{prefix}.weight_scale").to(torch.float32)
        w = q * scale.repeat_interleave(GROUP, dim=1)
        shape = self.get(f"{prefix}.weight_shape").tolist()
        if list(w.shape) != list(shape):
            w = w.reshape(shape)
        return w


def expert_forward(sh, prefix, x, weights=None):
    """x: [n, DIM] float32；weights: [n,1] float32，乘在激活上（照抄参考实现）。"""
    w1 = sh.dequant(f"{prefix}.w1")
    w3 = sh.dequant(f"{prefix}.w3")
    w2 = sh.dequant(f"{prefix}.w2")
    gate = x @ w1.t()
    up = x @ w3.t()
    if SWIGLU_LIMIT > 0:
        up = torch.clamp(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
        gate = torch.clamp(gate, max=SWIGLU_LIMIT)
    h = torch.nn.functional.silu(gate) * up
    if weights is not None:
        h = weights * h
    return h @ w2.t()


def stats(name, a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    diff = (a - b).abs()
    cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-30)
    print(f"  {name:34s} rms_a={a.pow(2).mean().sqrt():.4f} rms_b={b.pow(2).mean().sqrt():.4f} "
          f"maxdiff={diff.max():.4g} reldiff={diff.norm()/(b.norm()+1e-30):.4%} cos={cos:.6f}",
          flush=True)
    return cos.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/dump/dsv41_L0_moe.pt")
    ap.add_argument("--model", default="/models")
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    d = torch.load(args.dump, map_location="cpu")
    x_in = d["ffn_in"].float()          # vLLM 进入 MoE 的输入（已过 ffn_norm）
    x_vllm = d["x"].float()             # vLLM 的 FFN 输出
    print(f"  raw shapes: ffn_in={tuple(d['ffn_in'].shape)} x={tuple(d['x'].shape)}", flush=True)
    # mHC(hc_mult=4) 下张量可能是 [T, 4, DIM]；MoE 是逐行运算，摊平成 [T*4, DIM] 即可对拍
    if x_in.dim() == 3:
        x_in = x_in.reshape(-1, x_in.shape[-1])
    if x_vllm.dim() == 3:
        x_vllm = x_vllm.reshape(-1, x_vllm.shape[-1])
    assert x_in.shape[-1] == DIM, f"最后一维应为 {DIM}，实际 {x_in.shape}"
    assert x_in.shape == x_vllm.shape, f"输入/输出形状不一致: {x_in.shape} vs {x_vllm.shape}"
    T = x_in.shape[0]
    print(f"=== dump: T={T} layer={args.layer} "
          f"in_rms={x_in.pow(2).mean().sqrt():.4f} out_rms={x_vllm.pow(2).mean().sqrt():.4f} "
          f"out_absmax={x_vllm.abs().max():.4g} ===", flush=True)

    sh = Shards(args.model)
    L = args.layer

    # ---- Gate（照抄参考实现）----
    Wg = sh.get(f"layers.{L}.ffn.gate.weight").to(torch.float32)
    bias = sh.get(f"layers.{L}.ffn.gate.bias").to(torch.float32)
    scores = (x_in @ Wg.t())                      # /gate_temp, gate_temp=1.0
    scores = torch.nn.functional.softplus(scores).sqrt()
    idx = (scores + bias).topk(TOPK, dim=-1)[1]
    w = scores.gather(1, idx)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    w = w * ROUTE_SCALE
    print(f"  路由 top-6 (token0): {idx[0].tolist()}  权重和={w.sum(-1)[0].item():.4f} "
          f"(应=1.5)", flush=True)

    # ---- 逐专家累加（fp32）----
    y_routed = torch.zeros_like(x_in)
    uniq = idx.flatten().unique().tolist()
    print(f"  命中专家数={len(uniq)}（T={T} × top6 = {T*TOPK}）", flush=True)
    for e in uniq:
        pos, top = torch.where(idx == e)
        oe = expert_forward(sh, f"layers.{L}.ffn.experts.{e}", x_in[pos],
                            w[pos, top, None])
        y_routed[pos] += oe

    # ---- shared expert（不加权，直接相加）----
    y_shared = expert_forward(sh, f"layers.{L}.ffn.shared_experts", x_in)

    y_full = y_routed + y_shared
    print(f"  rms: routed={y_routed.pow(2).mean().sqrt():.4f} "
          f"shared={y_shared.pow(2).mean().sqrt():.4f} "
          f"full={y_full.pow(2).mean().sqrt():.4f}", flush=True)

    # ---- 对拍 ----
    print("\n--- 与 vLLM 的 FFN 输出对比 ---", flush=True)
    stats("参考(routed+shared) vs vLLM", y_full, x_vllm)
    stats("参考(仅 routed)     vs vLLM", y_routed, x_vllm)
    stats("参考(仅 shared)     vs vLLM", y_shared, x_vllm)
    print("\n判读：若『仅 routed』明显比『routed+shared』更接近 vLLM ⇒ shared expert 没被加上（根因）。",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
