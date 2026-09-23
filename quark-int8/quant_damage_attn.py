#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量"注意力投影 int4 化"造成的功能损伤（layer 0/2，真请求 dump）。

## 动机

v67 的分解显示：**注意力贡献占该层输出的 57.3%**（layer0）/ 52.6%（layer2），
而它的 5 个投影仍是 int4（~10% 权重误差）。FFN 修好后（layer0 MoE 偏差 2.05%），
注意力很可能成为**剩余损伤的主源**——但那是估算（"输出误差≈权重误差"），本脚本把它测实。

## 做法

复用 `ref_layer0_attn.py` 已验证的 layer-0 注意力参考（曾 vs vLLM cos 0.999279），
**rope / fp8 行量化 / sink / 输出投影分组全部逐字照抄那版**，只把权重来源做成可切换：
    (a) 源 fp8 反量化 = 理想
    (b) 现役 CT-int4
同一 `attn_in`、同一 positions，比两者输出。同时把 (b) 与 vLLM 实际输出对比做**量具自检**。

  输出 int4 vs 源   = 量化损伤（要量的）
  输出 int4 vs vLLM = 量具自检，应仍在 0.9993 量级（历史 0.999279）
"""
import json
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, "/w/quark-int8")
from convert_dsv41_ct_int4 import dequant_fp8_block  # noqa: E402

MODEL = "/models"
SRC = "/src"
NEW = "/new"
G = 32


def wmap(root):
    return json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]


def get(root, name, wm, cast=None):
    """按索引打开**正确的**分片（q_norm/kv_norm/attn_sink 未必同片）。"""
    with safe_open(f"{root}/{wm[name]}", framework="pt") as f:
        t = f.get_tensor(name)
    return t if cast is None else t.to(cast)


def deq_src(name, wm):
    with safe_open(f"{SRC}/{wm[f'{name}.weight']}", framework="pt") as f:
        w = f.get_tensor(f"{name}.weight")
    with safe_open(f"{SRC}/{wm[f'{name}.scale']}", framework="pt") as f:
        s = f.get_tensor(f"{name}.scale")
    return dequant_fp8_block(w, s, dtype=torch.float32)


def deq_any(root, name, wm):
    """从任意目录取权重：有 weight_packed 走 int4，否则读 bf16 weight（新策略的产物）。"""
    if f"{name}.weight_packed" not in wm:
        with safe_open(f"{root}/{wm[f'{name}.weight']}", framework="pt") as f:
            return f.get_tensor(f"{name}.weight").to(torch.float32)
    with safe_open(f"{root}/{wm[f'{name}.weight_packed']}", framework="pt") as f:
        pk = f.get_tensor(f"{name}.weight_packed").to(torch.int32)
    with safe_open(f"{root}/{wm[f'{name}.weight_scale']}", framework="pt") as f:
        sc = f.get_tensor(f"{name}.weight_scale").to(torch.float32)
    with safe_open(f"{root}/{wm[f'{name}.weight_shape']}", framework="pt") as f:
        shp = f.get_tensor(f"{name}.weight_shape").tolist()
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    q = ((pk.unsqueeze(-1) >> sh) & 0xF).reshape(pk.shape[0], -1).to(torch.float32) - 8.0
    return (q * sc.repeat_interleave(G, dim=1)).reshape(shp)


def deq_int4(name, wm):
    with safe_open(f"{MODEL}/{wm[f'{name}.weight_packed']}", framework="pt") as f:
        pk = f.get_tensor(f"{name}.weight_packed").to(torch.int32)
    with safe_open(f"{MODEL}/{wm[f'{name}.weight_scale']}", framework="pt") as f:
        sc = f.get_tensor(f"{name}.weight_scale").to(torch.float32)
    with safe_open(f"{MODEL}/{wm[f'{name}.weight_shape']}", framework="pt") as f:
        shp = f.get_tensor(f"{name}.weight_shape").tolist()
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    q = ((pk.unsqueeze(-1) >> sh) & 0xF).reshape(pk.shape[0], -1).to(torch.float32) - 8.0
    return (q * sc.repeat_interleave(G, dim=1)).reshape(shp)


def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def apply_rope_gptj(x, cos, sin):
    a = x[..., 0::2].float(); b = x[..., 1::2].float()
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


def apply_rope_inv(x, cos, sin):
    a = x[..., 0::2].float(); b = x[..., 1::2].float()
    return torch.stack((a * cos + b * sin, -a * sin + b * cos), dim=-1).flatten(-2)


def fp8_row_quant(kv):
    amax = kv.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / 448.0
    q = (kv / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.float() * scale, scale


def run(W, qn, kn, sink, x, pos, hd, rd, nh, ng, olr, eps, theta):
    T = x.shape[0]
    inv = 1.0 / (theta ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
    ang = torch.outer(pos.float(), inv)
    cos, sin = ang.cos(), ang.sin()
    qr = rmsnorm((W["wq_a"] @ x.T).T, qn, eps)
    q = (W["wq_b"] @ qr.T).T.view(T, nh, hd)
    q = torch.cat([q[..., :-rd], apply_rope_gptj(q[..., -rd:], cos[:, None, :], sin[:, None, :])], dim=-1)
    kv = rmsnorm((W["wkv"] @ x.T).T, kn, eps)
    kv = torch.cat([kv[..., :-rd], apply_rope_gptj(kv[..., -rd:], cos, sin)], dim=-1)
    kv, _ = fp8_row_quant(kv)
    s = torch.einsum("thd,sd->ths", q, kv) * (hd ** -0.5)
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    s = s.masked_fill(~causal[:, None, :], float("-inf"))
    m = s.max(dim=-1, keepdim=True).values
    p = torch.nan_to_num(torch.exp(s - m), nan=0.0)
    den = p.sum(-1, keepdim=True) + torch.exp(sink[None, :, None] - m)
    o = torch.einsum("ths,sd->thd", p, kv) / den
    o = torch.cat([o[..., :-rd], apply_rope_inv(o[..., -rd:], cos[:, None, :], sin[:, None, :])], dim=-1)
    o = o.reshape(T, ng, -1)
    wa = W["wo_a"].view(ng, olr, -1)
    y = torch.einsum("tgd,grd->tgr", o, wa).reshape(T, -1)
    return y @ W["wo_b"].T


def dev(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item(), F.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0).item()


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    d_in = torch.load(f"/dump/dsv41_L{L}_in.pt", map_location="cpu", weights_only=False)
    d_out = torch.load(f"/dump/dsv41_L{L}_attn.pt", map_location="cpu", weights_only=False)
    # 兼容旧 dump：探针曾把键硬编码成 layer0_*（已修，但既有 dump 仍是旧标签）
    _k = f"layer{L}_attn_in" if f"layer{L}_attn_in" in d_in else "layer0_attn_in"
    if _k != f"layer{L}_attn_in":
        print(f"  ⚠️ 用了旧标签键 {_k}（该 dump 来自修复前的探针；文件名已表明它是 layer {L}）")
    x = d_in[_k].float()
    vllm = d_out[f"layer{L}_attn_out"].float()
    _kp = f"layer{L}_positions" if f"layer{L}_positions" in d_in else "layer0_positions"
    pos = d_in[_kp].reshape(-1)

    cfg = json.load(open(f"{MODEL}/config.json"))["text_config"]
    kw = dict(hd=cfg["head_dim"], rd=cfg["qk_rope_head_dim"], nh=cfg["num_attention_heads"],
              ng=cfg["o_groups"], olr=cfg["o_lora_rank"], eps=cfg["rms_norm_eps"],
              theta=cfg["rope_theta"])
    print(f"=== layer {L}: T={x.shape[0]} attn_in rms={x.pow(2).mean().sqrt():.5f} "
          f"vLLM out rms={vllm.pow(2).mean().sqrt():.5f} ===", flush=True)

    iwm = wmap(MODEL)
    qn = get(MODEL, f"layers.{L}.attn.q_norm.weight", iwm, torch.float32)
    kn = get(MODEL, f"layers.{L}.attn.kv_norm.weight", iwm, torch.float32)
    sink = get(MODEL, f"layers.{L}.attn.attn_sink", iwm, torch.float32)

    names = ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")
    Ws = {n: deq_src(f"layers.{L}.attn.{n}", wmap(SRC)) for n in names}
    Wi = {n: deq_int4(f"layers.{L}.attn.{n}", iwm) for n in names}
    for n in names:
        r, c = dev(Wi[n], Ws[n])
        print(f"  权重 {n:5s} int4 vs 源: rel={r:.3%} cos={c:.6f}", flush=True)

    out_s = run(Ws, qn, kn, sink, x, pos, **kw)
    out_i = run(Wi, qn, kn, sink, x, pos, **kw)
    r_si, c_si = dev(out_i, out_s)
    r_iv, c_iv = dev(out_i, vllm)
    print(f"\n=== 注意力输出 ===")
    print(f"  源(理想) rms={out_s.pow(2).mean().sqrt():.5f}   int4 rms={out_i.pow(2).mean().sqrt():.5f}")
    print(f"  ▶ 旧 int4 损伤 vs 源 : 相对偏差={r_si:.3%}  cos={c_si:.6f}")
    print(f"  量具自检 int4 vs vLLM : 相对偏差={r_iv:.3%}  cos={c_iv:.6f}（历史 0.999279）")

    nwm = wmap(NEW)
    Wn = {n: deq_any(NEW, f"layers.{L}.attn.{n}", nwm) for n in names}
    for n in names:
        r, c = dev(Wn[n], Ws[n])
        kind = "bf16(新)" if f"layers.{L}.attn.{n}.weight_packed" not in nwm else "int4"
        print(f"  新策略权重 {n:5s} [{kind:8s}] vs 源: rel={r:.3%} cos={c:.6f}", flush=True)
    out_n = run(Wn, qn, kn, sink, x, pos, **kw)
    r_sn, c_sn = dev(out_n, out_s)
    print(f"\n  ▶ 新策略损伤 vs 源 : 相对偏差={r_sn:.3%}  cos={c_sn:.6f}"
          f"   （旧 {r_si:.3%} ⇒ 改善 {r_si/max(r_sn,1e-9):.2f}×）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
