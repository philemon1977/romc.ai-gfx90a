#!/usr/bin/env python3
"""验证 mHC 的**旋转链**（layer-0 其它检查全都照不到的一段），全部取自**同一次真实前向**。

参考语义（inference/model.py 的 Block.forward）：
    residual = x                                   # 进入本 block 的 hc 流
    attn_pre/post/comb = hc_mixes(x, hc_attn_fn)   # 本层注意力算出的三组系数
    x = hc_pre(x, pre_mix)                         # ← 用**传进来的** pre_mix（第 0 层是 identity）
    x = attn_norm(x); x = attn(x)
    x = hc_post(x, residual, attn_post, attn_comb) # ← 注意力后的残差
    residual = x
    ffn_pre/post/comb = hc_mixes(x, hc_ffn_fn)
    x = hc_pre(x, attn_pre)                        # ← ★ 旋转：FFN 用**本层注意力的** pre
    x = ffn_norm(x); x = ffn(x)

三项判据（A/B 层内，C 跨层）：
  A) residual_after_attn == post*attn_out + einsum('ij,ih->jh', comb, residual_before)
  B) ffn_normed_in      == ffn_norm( hc_pre(residual_after_attn, attn_pre) )     ← 旋转本身
  C) attn_in            == attn_norm( hc_pre(residual_before, pre_mix_in) )      ← 上一层传下来的 pre_mix
"""
import json, torch
from safetensors import safe_open
CKPT="/models"; DEV="cuda"
wm=json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
cfg=json.load(open(f"{CKPT}/config.json"))["text_config"]; eps=cfg["rms_norm_eps"]
def ck(n):
    with safe_open(f"{CKPT}/{wm[n]}",framework="pt") as f: return f.get_tensor(n).to(DEV).float()
def rms(x,w):
    x=x.float(); return (x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps))*w
def rel(a,b): return (a-b).abs().max().item()/(b.abs().max().item()+1e-9)

for L in (0,2):
    try:
        d=torch.load(f"/tmp/dsv41_L{L}_rot.pt",map_location="cpu",weights_only=False)
    except Exception as e:
        print(f"layer {L}: 缺 /tmp/dsv41_L{L}_rot.pt（{e!r}）"); continue
    T=d["attn_in"].shape[0]
    g=lambda k: d[k].to(DEV).float() if d.get(k) is not None else None
    attn_in,attn_out=g("attn_in"),g("attn_out")
    post,comb=g("attn_post_mix"),g("attn_res_mix")
    r_before=g("residual_before")
    attn_pre,pre_in=g("attn_pre"),g("pre_mix_in")
    r_after,ffn_in=g("residual_after_attn"),g("ffn_normed_in")
    print(f"\n=== layer {L} (T={T}) ===")
    print(f"  residual_before{tuple(r_before.shape)} post{tuple(post.shape)} comb{tuple(comb.shape)} "
          f"pre_mix_in={'None(identity)' if pre_in is None else tuple(pre_in.shape)}")
    # A
    mixed=torch.einsum("ij,ih->jh", comb, r_before) if comb.dim()==2 else torch.einsum("...ij,...ih->...jh", comb, r_before)
    p = post if post.dim()==3 else post.unsqueeze(-1)
    mine=(p*attn_out.unsqueeze(-2) if p.dim()==3 else p*attn_out)+mixed
    print(f"  A) mhc_post: 相对={rel(mine.float(),r_after):.3e} {'✅' if rel(mine.float(),r_after)<1e-2 else '◆不一致◆'}")
    # B 旋转
    collapsed=(attn_pre.unsqueeze(-1)*r_after.float()).sum(dim=1) if attn_pre.dim()==2 else (attn_pre.unsqueeze(-1)*r_after.float()).sum(dim=1)
    myb=rms(collapsed, ck(f"layers.{L}.ffn_norm.weight"))
    print(f"  B) ffn 输入(hc_pre 用 attn_pre): 相对={rel(myb,ffn_in):.3e} {'✅' if rel(myb,ffn_in)<1e-2 else '◆不一致◆ ⇒ 旋转用错了 mix'}")
    # C 跨层：注意力输入用**传进来的** pre_mix
    if pre_in is None:
        mya=rms(r_before[:,0], ck(f"layers.{L}.attn_norm.weight"))
        tag="identity ⇒ copy0"
    else:
        coll=(pre_in.unsqueeze(-1)*r_before.float()).sum(dim=1)
        mya=rms(coll, ck(f"layers.{L}.attn_norm.weight"))
        tag="用上一层传下来的 pre_mix"
    print(f"  C) attn 输入({tag}): 相对={rel(mya,attn_in):.3e} {'✅' if rel(mya,attn_in)<1e-2 else '◆不一致◆'}")
