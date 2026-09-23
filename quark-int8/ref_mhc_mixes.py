#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mHC coefficients 对拍：用 checkpoint 的**权威算法**复算 ffn 的 pre/post/comb，与 vLLM dump 对比。

## 为什么要有这个脚本

到 v63 为止，mHC 只有两把**非权威**的尺子：
  1) 公式抄写比对：vLLM 的 kernels/mhc/torch.py 与 checkpoint 的 kernel.py 逐行一致（读代码，非测量）；
  2) A/B 旋转链自洽：residual_after_attn == post*attn_out + einsum(comb, residual_before)
     —— 但它用的是 **vLLM 自己产出的 post/comb**，只证明「拿去用的方式对」，
        **没有证明 post/comb 的数值本身对**。
  3) 历史 _ROT dump 全是 tok=1（解码步）抓的，且没有真实请求门。

本脚本补上缺的那一环：拿真实请求（T=22）的 `residual_before_ffn_post`
（= 计算 ffn mixes 时用的那份残差）与 checkpoint 的 `layers.0.hc_ffn_*` 参数，
按 kernel.py::hc_split_sinkhorn 的**逐字算法**复算，再与 vLLM 的 ffn_post_mix /
ffn_res_mix 比。

## 特别针对的一类错误

`hc_post_alpha`（checkpoint 里是硬编码的 2×）**不在 config.json 的 text_config 里**。
若 vLLM 拿到的不是 2.0，post 会整体差一个常数因子，而这种错
「A/B 自洽」永远抓不到（自洽只要求 post 用的一致），却足以毁掉残差混合。

## 权威算法（kernel.py:407-462，逐字照抄）
  pre[j]  = sigmoid(mixes[j]*scale[0] + base[j]) + eps
  post[j] = 2 * sigmoid(mixes[j+hc]*scale[1] + base[j+hc])
  comb[j,k] = mixes[j*hc+k+2hc]*scale[2] + base[j*hc+k+2hc]
  comb = softmax(comb, -1) + eps
  comb = comb / (comb.sum(-2) + eps)              # 先列归一化
  for _ in range(iters-1):
      comb = comb / (comb.sum(-1) + eps)          # 行
      comb = comb / (comb.sum(-2) + eps)          # 列
  其中 mixes = (x.flatten(1) @ fn.T) * rsqrt(x.square().mean(-1) + norm_eps)
"""
import argparse
import json
import sys

import torch
from safetensors import safe_open


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc=4, iters=20, eps=1e-6, post_alpha=2.0):
    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = torch.sigmoid(mixes[:, hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc]) * post_alpha
    comb = mixes[:, 2 * hc:].view(-1, hc, hc) * hc_scale[2] + hc_base[2 * hc:].view(1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def rel(a, b):
    return ((a - b).abs().norm() / (b.norm() + 1e-30)).item()


def load_params(model_dir, layer):
    with open(f"{model_dir}/model.safetensors.index.json") as fh:
        wmap = json.load(fh)["weight_map"]
    out = {}
    handles = {}
    for tag in ("hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
                "hc_attn_fn", "hc_attn_scale", "hc_attn_base"):
        name = f"layers.{layer}.{tag}"
        if name not in wmap:
            continue
        f = wmap[name]
        if f not in handles:
            handles[f] = safe_open(f"{model_dir}/{f}", framework="pt")
        out[tag] = handles[f].get_tensor(name).to(torch.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moe-dump", default="/dump/dsv41_L0_moe.pt")
    ap.add_argument("--model", default="/models")
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    cfg = json.load(open(f"{args.model}/config.json"))["text_config"]
    hc = int(cfg["hc_mult"])
    iters = int(cfg["hc_sinkhorn_iters"])
    eps = float(cfg["hc_eps"])
    norm_eps = float(cfg.get("rms_norm_eps", 1e-6))
    print(f"=== config: hc_mult={hc} sinkhorn_iters={iters} hc_eps={eps:g} "
          f"rms_norm_eps={norm_eps:g} ===", flush=True)

    d = torch.load(args.moe_dump, map_location="cpu")
    residual = d["residual_before_ffn_post"].float()      # [T, hc, D]
    v_post = d["ffn_post_mix"].float().squeeze(-1)        # [T, hc]
    v_comb = d["ffn_res_mix"].float()                     # [T, hc, hc]
    T = residual.shape[0]
    print(f"  dump: T={T} residual={tuple(residual.shape)} "
          f"v_post={tuple(v_post.shape)} v_comb={tuple(v_comb.shape)}", flush=True)

    p = load_params(args.model, args.layer)
    need = {"hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"}
    if not need <= set(p):
        print(f"❌ 缺参数: {need - set(p)}", flush=True)
        return 2

    x = residual.reshape(T, -1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = (x @ p["hc_ffn_fn"].t()) * rsqrt
    print(f"  mixes: {tuple(mixes.shape)}  absmax={mixes.abs().max():.4g}  "
          f"hc_ffn_scale={p['hc_ffn_scale'].tolist()}", flush=True)

    for alpha in (2.0, 1.0):
        pre, post, comb = hc_split_sinkhorn(mixes, p["hc_ffn_scale"], p["hc_ffn_base"],
                                            hc, iters, eps, post_alpha=alpha)
        r_post = rel(post, v_post)
        r_comb = rel(comb, v_comb)
        flag = "✅" if (r_post < 5e-3 and r_comb < 5e-3) else "◆不一致◆"
        print(f"  post_alpha={alpha}: post 相对={r_post:.3e}  comb 相对={r_comb:.3e}  {flag}",
              flush=True)

    pre, post, comb = hc_split_sinkhorn(mixes, p["hc_ffn_scale"], p["hc_ffn_base"],
                                        hc, iters, eps, post_alpha=2.0)
    print(f"\n  细节(post_alpha=2.0): v_post[0]={v_post[0].tolist()}", flush=True)
    print(f"                        参考[0]={post[0].tolist()}", flush=True)
    print(f"  v_comb[0]=\n{v_comb[0]}", flush=True)
    print(f"  参考comb[0]=\n{comb[0]}", flush=True)
    print(f"  行和 v={v_comb[0].sum(-1).tolist()} 列和 v={v_comb[0].sum(-2).tolist()}", flush=True)
    print(f"  行和 参考={comb[0].sum(-1).tolist()} 列和 参考={comb[0].sum(-2).tolist()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
