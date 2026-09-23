"""gfx90a 上实际走的 mhc 实现（tilelang）vs vLLM 自己的 torch 参考实现。

背景：checkpoint 自带 inference/kernel.py 里 hc_split_sinkhorn 的权威语义，
vLLM 的 torch 实现与之逐项一致（pre=sigmoid+eps；post=sigmoid*2.0；
comb=softmax(-1)+eps → /(colsum+eps) → 19 轮行列交替归一化；
delayed：layer_input 用**传入的** pre_mix 折叠，pre_mix=None 时取 copy 0）。
但 gfx90a 上 HAS_AITER_MHC=False、HAS_TILELANG_MHC=True ⇒ 实际执行的是
mhc_pre_delayed_tilelang / mhc_post_tilelang。上游自己给 gfx942 关了 tilelang
（注释：“until that path is fixed”），gfx90a 没关。
"""
import json, os, sys
import torch

MODEL = "/models"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

from safetensors import safe_open
wm = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
def ckpt(name, dtype=None):
    with safe_open(f"{MODEL}/{wm[name]}", framework="pt") as f:
        t = f.get_tensor(name)
    return t.to(dtype) if dtype else t

from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch, mhc_post_torch  # noqa

dev = "cuda:0"
torch.manual_seed(0)
HC, H, T = 4, 5120, 19
EPS = 1e-6
RMS_EPS = 1e-20

fn_full = ckpt("layers.0.hc_attn_fn", torch.float32).to(dev)      # [24, 20480]
hc_scale = ckpt("layers.0.hc_attn_scale", torch.float32).to(dev)  # [3]
hc_base = ckpt("layers.0.hc_attn_base", torch.float32).to(dev)    # [24]
print(f"fn{tuple(fn_full.shape)} scale={hc_scale.tolist()} base[min={hc_base.min():.4f} max={hc_base.max():.4f}]")

def cmp(tag, a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs().max().item()
    denom = b.abs().max().item() + 1e-9
    ok = d / denom < 1e-3
    print(f"    {tag:14s} maxabs={d:.3e}  相对={d/denom:.2e}  {'ok' if ok else '◆不一致◆'}")
    return ok

fails = 0
for case in ("ffn/后续层(pre_mix 传入)", "第一层(广播 fn + x=x)"):
    if case.startswith("第一层"):
        fn = fn_full.view(-1, HC, H).sum(1).contiguous()          # [24, 5120]
        xin = (torch.randn(T, H, device=dev) * 0.6).to(torch.bfloat16)
        residual = xin.unsqueeze(1).expand(-1, HC, -1).contiguous()
        pre_mix = None
    else:
        fn = fn_full
        xin = None
        residual = (torch.randn(T, HC, H, device=dev) * 0.6).to(torch.bfloat16)
        pre_mix = torch.softmax(torch.randn(T, HC, device=dev), dim=-1).contiguous()

    ref = mhc_pre_delayed_torch(residual, fn, hc_scale, hc_base, RMS_EPS, EPS, EPS,
                                2.0, 20, pre_mix, xin)
    got = torch.ops.vllm.mhc_pre_delayed_tilelang(residual, fn, hc_scale, hc_base,
                                                  RMS_EPS, EPS, EPS, 2.0, 20,
                                                  pre_mix, xin, None, 1e-6)
    print(f"  用例: {case}")
    names = ("post_mix", "comb_mix", "layer_input(=attn 输入)", "pre_mix(下一层用)")
    for n, a, b in zip(names, got, ref):
        if not cmp(n, a, b):
            fails += 1
            print(f"        tilelang[0,:8]={a.flatten()[:8].tolist()}")
            print(f"        torch   [0,:8]={b.flatten()[:8].tolist()}")

# mhc_post
residual = (torch.randn(T, HC, H, device=dev) * 0.6).to(torch.bfloat16)
xout = (torch.randn(T, H, device=dev) * 0.6).to(torch.bfloat16)
post = torch.sigmoid(torch.randn(T, HC, 1, device=dev))
comb = torch.softmax(torch.randn(T, HC, HC, device=dev), dim=-1)
ref = mhc_post_torch(xout, residual, post, comb)
got = torch.ops.vllm.mhc_post_tilelang(xout, residual, post, comb)
print("  用例: mhc_post")
if not cmp("mhc_post", got, ref):
    fails += 1
print(f"\n结论：{'全部一致 ✅（tilelang 与 torch 参考等价）' if fails==0 else f'{fails} 项不一致 ❌ —— gfx90a 走的 tilelang 路径与参考语义不符'}")
