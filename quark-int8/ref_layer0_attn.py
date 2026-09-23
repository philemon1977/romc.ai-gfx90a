#!/usr/bin/env python3
"""layer-0 注意力的**独立 torch 参考**（规格 = checkpoint 自带 inference/model.py + kernel.py）。

对拍对象：v53 从 vLLM 真实前向里 dump 出来的
  /tmp/dsv41_L0_in.pt   {layer0_attn_in [T,5120] bf16, layer0_positions, layer0_input_ids}
  /tmp/dsv41_L0_attn.pt {layer0_attn_in, layer0_attn_out [T,5120] bf16}

层 0 的 compress_ratios[0]=0 ⇒ 纯滑窗、无压缩/无索引器、无 YaRN（参考注释："disable YaRN
and use base rope_theta in pure sliding-window attention"）⇒ 这一层是**唯一**能纯 torch 复刻的层。

参考语义（逐条对照 inference/model.py / kernel.py）：
  qr = q_norm(wq_a(x))                       # RMSNorm(1280, eps=1e-20)
  q  = wq_b(qr).view(T, 64, 512)
  apply_rotary_emb(q[..., -64:], freqs)      # GPT-J 相邻对，theta=10000，无 yarn
  kv = kv_norm(wkv(x))[..., 512]             # 单 KV 头（K 与 V 同源）
  apply_rotary_emb(kv[..., -64:], freqs)
  act_quant(kv, ...)                         # 每**行**一个 fp32 scale（512B fp8 + 4B）
  o = sparse_attn(q, kv, attn_sink, causal, scale=head_dim**-0.5)
      # kernel.py:84  sum_exp += exp(attn_sink[i] - scores_max[i])  ⇒ sink 是"加在分母上的 logit"
  apply_rotary_emb(o[..., -64:], freqs, inverse=True)
  o  = o.view(T, 8, 4096);  y = einsum('tgd,grd->tgr', o, wo_a.view(8,1024,4096));  out = y.flatten(1) @ wo_b.T
"""
import json, math, sys
import torch

MODEL = "/models"
DEV = "cuda:0"
torch.manual_seed(0)

from safetensors import safe_open
wm = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]

def raw(name):
    with safe_open(f"{MODEL}/{wm[name]}", framework="pt") as f:
        return f.get_tensor(name)

def deq(name, group=32):
    """CT/uint4b8：真值 = (nibble - 8) * scale；checkpoint 布局 [out, in/8] I32 + [out, in/G]。"""
    wp = raw(f"{name}.weight_packed").to(DEV)
    ws = raw(f"{name}.weight_scale").to(DEV).float()
    shifts = torch.arange(8, device=DEV, dtype=torch.int32) * 4
    nib = ((wp.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF).reshape(wp.shape[0], -1)
    return ((nib.to(torch.float32) - 8.0) * ws.repeat_interleave(group, dim=1))

def plain(name, dtype=None):
    t = raw(name).to(DEV)
    return t.to(dtype) if dtype else t

def rmsnorm(x, w, eps):
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * w.float()

def apply_rope_gptj(x, cos, sin):
    """x: [..., d]，d 为偶数；GPT-J 相邻对。cos/sin: [d/2]"""
    a = x[..., 0::2].float(); b = x[..., 1::2].float()
    o = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)
    return o.flatten(-2)

def apply_rope_inv(x, cos, sin):
    a = x[..., 0::2].float(); b = x[..., 1::2].float()
    o = torch.stack((a * cos + b * sin, -a * sin + b * cos), dim=-1)
    return o.flatten(-2)

def fp8_row_quant(kv):
    """参考 act_quant：整行（512 维）一个 scale；e4m3 上限 448。"""
    amax = kv.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / 448.0
    q = (kv / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.float() * scale, scale

def main():
    d_in = torch.load("/tmp/dsv41_L0_in.pt", map_location="cpu", weights_only=False)
    d_out = torch.load("/tmp/dsv41_L0_attn.pt", map_location="cpu", weights_only=False)
    x = d_in["layer0_attn_in"].to(DEV).float()
    ref_vllm = d_out["layer0_attn_out"].to(DEV).float()
    pos = d_in["layer0_positions"].to(DEV).reshape(-1)
    T = x.shape[0]
    print(f"T={T} positions={pos.tolist()[:24]}  vLLM attn_out absmax={ref_vllm.abs().max():.4g}")

    cfg = json.load(open(f"{MODEL}/config.json"))["text_config"]
    eps = cfg["rms_norm_eps"]; hd = cfg["head_dim"]; rd = cfg["qk_rope_head_dim"]
    nh = cfg["num_attention_heads"]; ng = cfg["o_groups"]; olr = cfg["o_lora_rank"]
    theta = cfg["rope_theta"]
    print(f"eps={eps} head_dim={hd} rope_dim={rd} heads={nh} groups={ng} o_lora={olr} theta={theta}")

    W = {n: deq(f"layers.0.attn.{n}") for n in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")}
    qn = plain("layers.0.attn.q_norm.weight", torch.float32)
    kn = plain("layers.0.attn.kv_norm.weight", torch.float32)
    sink = plain("layers.0.attn.attn_sink", torch.float32)
    print("权重:", {k: tuple(v.shape) for k, v in W.items()}, "sink", tuple(sink.shape))

    # RoPE：层 0 无 yarn（original_seq_len=0）⇒ 纯 theta=10000
    inv_freq = 1.0 / (theta ** (torch.arange(0, rd, 2, device=DEV, dtype=torch.float32) / rd))
    ang = torch.outer(pos.float(), inv_freq)
    cos, sin = ang.cos(), ang.sin()

    qr = rmsnorm((W["wq_a"] @ x.T).T, qn, eps)                     # [T,1280]
    q = (W["wq_b"] @ qr.T).T.view(T, nh, hd)
    q = torch.cat([q[..., :-rd], apply_rope_gptj(q[..., -rd:], cos[:, None, :], sin[:, None, :])], dim=-1)

    kv = rmsnorm((W["wkv"] @ x.T).T, kn, eps)                      # [T,512]
    kv = torch.cat([kv[..., :-rd], apply_rope_gptj(kv[..., -rd:], cos, sin)], dim=-1)
    kv, kscale = fp8_row_quant(kv)
    print(f"kv fp8 scale: min={kscale.min():.3e} max={kscale.max():.3e}  kv_deq absmax={kv.abs().max():.4g}")

    scale = hd ** -0.5
    s = torch.einsum("thd,sd->ths", q, kv) * scale                 # [T,heads,T]
    causal = torch.ones(T, T, device=DEV, dtype=torch.bool).tril()
    s = s.masked_fill(~causal[:, None, :], float("-inf"))
    m = s.max(dim=-1, keepdim=True).values
    p = torch.exp(s - m)
    p = torch.nan_to_num(p, nan=0.0)
    den = p.sum(dim=-1, keepdim=True) + torch.exp(sink[None, :, None] - m)
    o = torch.einsum("ths,sd->thd", p, kv) / den
    o = torch.cat([o[..., :-rd], apply_rope_inv(o[..., -rd:], cos[:, None, :], sin[:, None, :])], dim=-1)

    o = o.reshape(T, ng, -1)
    wa = W["wo_a"].view(ng, olr, -1)
    y = torch.einsum("tgd,grd->tgr", o, wa).reshape(T, -1)
    out = y @ W["wo_b"].T

    a, b = out, ref_vllm
    rel = (a - b).abs().mean() / (b.abs().mean() + 1e-9)
    cos_s = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    print(f"\n=== 参考 vs vLLM（layer 0 注意力输出）===")
    print(f"  参考 absmax={a.abs().max():.4g}  vLLM absmax={b.abs().max():.4g}")
    print(f"  相对误差 = {rel*100:.3f}%   cos = {cos_s:.6f}   最大绝对差 = {(a-b).abs().max():.4g}")
    print(f"  判据：cos>0.999 且 相对误差<2% ⇒ 注意力（含 rope/量化/sink/输出投影）正确")
    # 逐元素前 8 个，便于肉眼确认不是尺度问题
    print(f"  参考[0,:6] = {[round(float(v),4) for v in a[0,:6]]}")
    print(f"  vLLM[0,:6] = {[round(float(v),4) for v in b[0,:6]]}")

main()
