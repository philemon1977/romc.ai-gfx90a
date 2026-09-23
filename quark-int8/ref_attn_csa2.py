#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CSA2 单层端到端对拍（layer 2 = 首个 Full 模式层）。

## 为什么是 layer 2、为什么这个方法可行

此前"注意力已验证"其实只覆盖了 layer 0 —— 而 layer 0 的 compress_ratio=0，走的是
**SWA-only** 分支（40 层里只占 2 层）。CSA2（Full/Reindex/Reuse + 跨层 KV/index 复用）
覆盖 38/40 层，**从未做过端到端对拍**。

外部建议的"四象限"里 reference 两格在 gfx90a 上跑不了（官方 `inference/model.py`
依赖 tilelang 的 fp8/fp4 内核）。可行的替代是把已验证有效的方法下沉为
**单层 + 真实输入**：拿 vLLM 自己的层输入，按官方公式复算该层输出，逐张量比。
这正是"张量级 state 对拍"——权重级测试覆盖不到，张量级可以 ✓。

选 layer 2 的理由：它同时是 kv_source 与 index_source（自给自足，不依赖更早的共享状态 ✓），
T=22、ratio=2 ⇒ 压缩条目只有 11 个、`index_topk=512 ≫ 11` ⇒ DSA 退化为全选 ✓，
于是这一层可以纯 torch 完整复算。

## 官方公式（inference/model.py 逐字）

    qr = q_norm(wq_a(x)); q = wq_b(qr).view(T,nh,hd); rope(q[...,-rd:])
    kv = kv_norm(wkv(x)); rope; act_quant(kv, 32, ue8m0, E8M0, inplace=True)   # 窗口 K：fp8 fake-quant
    latent = compressor(x); latent[...,-rd:] rope(位置取 group 首 token j*ratio)
    fp4_act_quant(latent, 16, inplace=True, scale_dtype=E4M3)                   # 压缩 KV：fp4 fake-quant
    kv_all = cat([窗口22, 压缩11]); topk = cat([窗口idx, 压缩idx])
    o = sparse_attn(q, kv_all, attn_sink, topk, scale=hd**-0.5)
    o = rope_inv(o); o.view(T,ng,-1) @wo_a(分组) → wo_b

## 两个未知量，用开关穷举反推运行时行为

1. 压缩 KV 是否做 fp4 fake-quant（官方做；我们持久 cache 是 bf16 ⇒ 可能跳过）
2. 压缩条目的可见性规则：`2j ≤ p`（组视为位于首 token ✓ 官方注释）还是 `2j+1 ≤ p`
   ⇒ 两种都跑，看哪个更贴 vLLM —— **用实验钉死规则，而不是猜** ✓
"""
import argparse
import itertools
import math
import json
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, "/w/quark-int8")

FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def wmap(d):
    return json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]


class W:
    """从 CT-Int4 checkpoint 读权重：packed 走 int4 反量化，否则直接读 dense。"""

    def __init__(self, d):
        self.d = d
        self.m = wmap(d)
        self.h = {}
        sys.path.insert(0, "/w/quark-int8")
        import convert_dsv41_ct_int4 as C
        self.C = C

    def _f(self, shard):
        if shard not in self.h:
            self.h[shard] = safe_open(f"{self.d}/{shard}", framework="pt")
        return self.h[shard]

    def get(self, name):
        return self._f(self.m[name]).get_tensor(name)

    def lin(self, mod):
        if f"{mod}.weight_packed" in self.m:
            pk = self.get(f"{mod}.weight_packed").to(torch.int32)
            sc = self.get(f"{mod}.weight_scale").to(torch.float32)
            shp = self.get(f"{mod}.weight_shape").tolist()
            sh = torch.arange(0, 32, 4, dtype=torch.int32)
            q = ((pk.unsqueeze(-1) >> sh) & 0xF).reshape(pk.shape[0], -1).to(torch.float32) - 8.0
            return (q * sc.repeat_interleave(32, dim=1)).reshape(shp)
        return self.get(f"{mod}.weight").to(torch.float32)


def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_gptj(x, cos, sin, inverse=False):
    a, b = x[..., 0::2], x[..., 1::2]
    if inverse:
        return torch.stack((a * cos + b * sin, -a * sin + b * cos), dim=-1).flatten(-2)
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """★ 官方纯 torch 实现（inference/model.py:369）逐字照抄 —— 不再用 vLLM 的 rope，
    否则"vLLM 的 rope 是否正确"就成了循环论证。"""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def cis_to_cos_sin(cis, pos):
    """把官方 complex 频率表取到给定位置，转成与 ref_layer0_attn 相同的 (cos, sin) 用法。"""
    c = cis[pos].real.contiguous()
    sn = cis[pos].imag.contiguous()
    return c, sn


def fake_fp8_ue8m0(x, blk=32):
    """官方 _window_kv 的 act_quant(kv,32,'ue8m0',E8M0,inplace=True)：
    scale 取 2 的幂（e8m0），值量化到 e4m3 再反量化回 bf16 量级。"""
    *lead, n = x.shape
    v = x.reshape(-1, n // blk, blk).double()
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / 448.0))          # UE8M0：scale 必须是 2 的幂
    s = torch.pow(2.0, e)
    q = (v / s).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (q.double() * s).reshape(*lead, n).to(torch.float32)


def fake_fp4_e4m3(x, blk=16):
    """官方 _compress_kv 的 fp4_act_quant(latent,16,inplace=True,E4M3)：
    e2m1 栅格 + 每组 16 一个 **E4M3** scale（不是 e8m0！）。"""
    *lead, n = x.shape
    v = x.reshape(-1, n // blk, blk).double()
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    s = (amax / 6.0)
    s = s.to(torch.float8_e4m3fn).double()            # scale 本身落到 E4M3
    s = s.clamp_min(1e-30)
    lut = FP4_LEVELS.double()
    scaled = (v / s).abs().clamp(0, 6.0)
    idx = torch.stack([scaled - lut[i] for i in range(8)], -1).abs().argmin(-1)
    q = lut[idx] * torch.sign(v)
    return (q * s).reshape(*lead, n).to(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/dump/dsv41_L2_attn.pt")
    ap.add_argument("--in-dump", default="/dump/dsv41_L2_in.pt")
    ap.add_argument("--cmp-in", default="/dump/dsv41_cmp_in.pt")
    ap.add_argument("--cmp-out", default="/dump/dsv41_cmp_out.pt")
    ap.add_argument("--model", default="/models")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--compress-blk", type=int, default=16)
    ap.add_argument("--fp4", type=int, default=1, help="1=做 fp4 fake-quant, 0=跳过（bf16 直通）")
    ap.add_argument("--fp8win", type=int, default=1, help="1=窗口K做 fp8 fake-quant")
    ap.add_argument("--vis", default="first", help="first: 2j<=p ; last: 2j+1<=p")
    a = ap.parse_args()

    L = a.layer
    d_out = torch.load(a.dump, map_location="cpu", weights_only=False)
    k = f"layer{L}_attn_out"
    if k not in d_out:
        k = "layer0_attn_out"
    vllm = d_out[k].float()
    x = d_out[f"layer{L}_attn_in"].float() if f"layer{L}_attn_in" in d_out else d_out["layer0_attn_in"].float()
    T = x.shape[0]
    din = torch.load(a.in_dump, map_location="cpu", weights_only=False)
    pos = din.get(f"layer{L}_positions", din.get("layer0_positions")).reshape(-1)[:T]
    print(f"layer {L}: T={T} vLLM out rms={vllm.pow(2).mean().sqrt():.5f}", flush=True)

    Wl = W(a.model)
    cfg = json.load(open(f"{a.model}/config.json"))["text_config"]
    hd, rd = cfg["head_dim"], cfg["qk_rope_head_dim"]
    nh, ng, olr = cfg["num_attention_heads"], cfg["o_groups"], cfg["o_lora_rank"]
    eps = cfg["rms_norm_eps"]
    ratio = cfg["compress_ratios"][L]
    use_yarn = ratio > 0
    rs = cfg.get("rope_scaling") or {}
    print(f"  ratio={ratio} head_dim={hd} rd={rd} heads={nh} groups={ng} yarn={use_yarn} "
          f"factor={rs.get('factor')} orig={rs.get('original_max_position_embeddings')}", flush=True)

    # 官方规则（model.py:681-687）：ratio>0 用 (original_seq_len, compress_rope_theta)，
    #                                    ratio=0 用 original_seq_len=0 + rope_theta
    dim_rope = rd
    seqlen = int(cfg.get("max_position_embeddings", 1048576))
    if use_yarn:
        orig, base = int(cfg.get("original_max_position_embeddings", 65536)), cfg["compress_rope_theta"]
    else:
        orig, base = 0, cfg["rope_theta"]
    cis = precompute_freqs_cis(dim_rope, min(seqlen, 4096), orig, base,
                               float(rs.get("factor", 16)), float(rs.get("beta_fast", 32)),
                               float(rs.get("beta_slow", 1)))
    cos, sin = cis_to_cos_sin(cis, pos.long())
    print(f"  rope: orig_seq_len={orig} base={base} factor={rs.get('factor')} "
          f"beta={rs.get('beta_fast')}/{rs.get('beta_slow')} ⇒ {'YaRN' if orig>0 else 'plain'}", flush=True)

    qn = Wl.get(f"layers.{L}.attn.q_norm.weight").float()
    kn = Wl.get(f"layers.{L}.attn.kv_norm.weight").float()
    sink = Wl.get(f"layers.{L}.attn.attn_sink").float()
    qr = rmsnorm(x @ Wl.lin(f"layers.{L}.attn.wq_a").t(), qn, eps)
    q = (qr @ Wl.lin(f"layers.{L}.attn.wq_b").t()).view(T, nh, hd)
    q = torch.cat([q[..., :-rd], rope_gptj(q[..., -rd:], cos[:, None, :], sin[:, None, :])], -1)
    kv = rmsnorm(x @ Wl.lin(f"layers.{L}.attn.wkv").t(), kn, eps)
    kv = torch.cat([kv[..., :-rd], rope_gptj(kv[..., -rd:], cos, sin)], -1)
    if a.fp8win:
        kv = fake_fp8_ue8m0(kv, 32)

    # 压缩条目：优先用探针 dump 的 latent（省掉重复实现 compressor）
    # 官方 Compressor.forward（model.py:458-487）逐字：ratio==1 ⇒ norm(wkv(x))（无池化无门）；
    # ratio>1 ⇒ kv,wgate 各 [T,hd]，按组做 softmax(kv 加权)，再 RMSNorm。
    # 这样就不依赖 DSV41_CMP_DEBUG 的 dump（那探针只记录首次触发，未必是 layer2）✓
    base_c = f"layers.{L}.attn.compressor"
    wkv_c = Wl.lin(f"{base_c}.wkv"); wgate_c = Wl.lin(f"{base_c}.wgate")
    cn = Wl.get(f"{base_c}.norm.weight").float()
    xf = x.float()
    if ratio == 1:
        lat = rmsnorm(xf @ wkv_c.t(), cn, eps)
        print(f"  compressor: ratio=1 ⇒ norm(wkv(x)) 无池化  latent{tuple(lat.shape)}", flush=True)
    else:
        kv_c = xf @ wkv_c.t(); sc_c = xf @ wgate_c.t()
        rem = T % ratio; cut = T - rem
        assert cut > 0, f"T={T} < ratio={ratio}，本层无完整组，需换层测"
        g_kv = kv_c[:cut].unflatten(0, (-1, ratio)); g_sc = sc_c[:cut].unflatten(0, (-1, ratio))
        lat = (g_kv * g_sc.softmax(dim=1)).sum(dim=1)
        lat = rmsnorm(lat, cn, eps)
        print(f"  compressor: ratio={ratio} T={T} 余数={rem} ⇒ 完整组={lat.shape[0]} "
              f"latent{tuple(lat.shape)}", flush=True)
    C = min(T // ratio, lat.shape[0])
    lat = lat[:C]
    cpos = pos[:C * ratio : ratio]
    cc, cs = cos[cpos], sin[cpos]
    lat = torch.cat([lat[..., :-rd], rope_gptj(lat[..., -rd:], cc, cs)], -1)
    if a.fp4:
        lat = fake_fp4_e4m3(lat, a.compress_blk)

    kv_all = torch.cat([kv, lat], 0)                       # [T+C, 512]
    N = kv_all.shape[0]
    s = (q @ kv_all.t()) * (hd ** -0.5)                     # [T,heads,N]
    mask = torch.zeros(T, N, dtype=torch.bool)
    mask[:, :T] = torch.ones(T, T, dtype=torch.bool).tril()  # 窗口：标准因果
    for j in range(C):
        key = 2 * j if a.vis == "first" else 2 * j + 1
        col = T + j
        mask[:, col] = (pos >= key)
    s = s.masked_fill(~mask[:, None, :], float("-inf"))
    m = s.amax(-1, keepdim=True)
    p = torch.nan_to_num(torch.exp(s - m), nan=0.0)
    den = p.sum(-1, keepdim=True) + torch.exp(sink[None, :, None] - m)
    o = (p @ kv_all) / den
    o = torch.cat([o[..., :-rd], rope_gptj(o[..., -rd:], cos[:, None, :], sin[:, None, :], True)], -1)
    wo_a = Wl.lin(f"layers.{L}.attn.wo_a").view(ng, olr, -1)
    y = torch.einsum("tgd,grd->tgr", o.view(T, ng, -1), wo_a).reshape(T, -1)
    out = y @ Wl.lin(f"layers.{L}.attn.wo_b").t()

    a64, b64 = out.double().flatten(), vllm.double().flatten()
    rel = ((a64 - b64).norm() / b64.norm()).item()
    cosv = F.cosine_similarity(out.flatten(), vllm.flatten(), dim=0).item()
    print(f"\n  fp4={a.fp4} fp8win={a.fp8win} vis={a.vis} cb={a.compress_blk} "
          f"⇒  相对偏差={rel:8.4%}  cos={cosv:.6f}  "
          f"{'✅ 命中' if cosv > 0.999 else ''}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
