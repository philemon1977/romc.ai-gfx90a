#!/usr/bin/env python3
"""压缩器输入投影（`compressor.fused_wkv_wgate`）的独立参考。

vLLM 把压缩器的 wkv/wgate **融成一个模块** `compressor.fused_wkv_wgate`
（shard0=wkv, shard1=wgate ⇒ 输出布局 [kv | score]），而 checkpoint 里是两个
独立张量 `attn.compressor.wkv` / `attn.compressor.wgate`（在 CT ignore 名单里，未量化）。
本脚本用 dump 的 kv_score 与同一层的 attn 输入 x，按 checkpoint 权重复算两个分支，
逐半比对——可抓"融合装载错位/交换"这类错。
"""
import json, torch
from safetensors import safe_open
CKPT="/models"; DEV="cuda"
wm=json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
def ck(n,d=None):
    with safe_open(f"{CKPT}/{wm[n]}",framework="pt") as f: t=f.get_tensor(n)
    return t.to(d) if d else t

d=torch.load("/tmp/dsv41_cmp_in.pt",map_location="cpu",weights_only=False)
kvs=d["kv_score"].to(DEV).float(); hd=int(d["head_dim"]); T=kvs.shape[0]
print(f"kv_score{tuple(kvs.shape)} head_dim={hd}")
_d=torch.load("/tmp/dsv41_L2_rot.pt",map_location="cpu",weights_only=False)
x=_d["attn_in"].to(DEV).float()          # layer-2 注意力输入 = 压缩器投影的输入
print(f"layer2 注意力输入 x{tuple(x.shape)}（即压缩器的输入）")
assert x.shape[0]==T, f"token 数不一致: x {x.shape[0]} vs kv_score {T}"

for L in (2,):
    try:
        wkv=ck(f"layers.{L}.attn.compressor.wkv.weight", torch.float32).to(DEV)
        wgate=ck(f"layers.{L}.attn.compressor.wgate.weight", torch.float32).to(DEV)
    except Exception as e:
        print(f"layer {L}: 取权重失败 {e!r}"); continue
    print(f"layer {L}: wkv{tuple(wkv.shape)} wgate{tuple(wgate.shape)} dtype(fp32 未量化)")
    kv_ref = x @ wkv.t(); score_ref = x @ wgate.t()
    for name, ref, got in (("[kv|score] 前半 = wkv(x)", kv_ref, kvs[:, :hd]),
                           ("[kv|score] 后半 = wgate(x)", score_ref, kvs[:, hd:])):
        dif=(ref-got).abs().max().item(); rel=dif/(got.abs().max().item()+1e-9)
        print(f"   {name}: maxabs={dif:.4e} 相对={rel:.3e} "
              f"{'✅ 一致' if rel<5e-3 else '◆不一致◆'}")
    # 交叉：若两半被交换，下面这条会同时"另一条不成立"
    sw=(kvs[:, :hd]-score_ref).abs().max().item()/(score_ref.abs().max().item()+1e-9)
    print(f"   （对照）前半 vs wgate(x) 相对差={sw:.3e} —— 若与上面同时很小则分不清，否则可判是否交换")
