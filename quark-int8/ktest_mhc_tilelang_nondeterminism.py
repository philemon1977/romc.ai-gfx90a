import json, os, torch
MODEL="/models"; os.environ.setdefault("VLLM_LOGGING_LEVEL","ERROR")
from safetensors import safe_open
wm=json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
# 前缀自适应：不同 checkpoint 的 mHC 张量前缀不同（本机 GLM-5.3-Flash-Quark-Int8 是
# "model.language_model."；09-18 初版脚本按无前缀写死 ⇒ 直接 KeyError）。45 层 ×(attn,ffn)
# ×(fn,scale,base) = 270 个 hc_* 张量，可当自检。
_hit=[k for k in wm if k.endswith("layers.0.hc_attn_fn")]
assert _hit, "index 里没有 layers.0.hc_attn_fn"
PFX=_hit[0][: -len("layers.0.hc_attn_fn")]
_hc=sum(1 for k in wm if ".hc_" in k)
print(f"mHC 前缀={PFX!r}  hc_* 张量数={_hc}（期望 45×6=270）")
def ckpt(n,d=None):
    with safe_open(f"{MODEL}/{wm[n]}",framework="pt") as f: t=f.get_tensor(n)
    return t.to(d) if d else t
from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch
dev="cuda:0"; T=19; EPS=1e-6; RMS=1e-20
# HC/H 必须来自 checkpoint 自己的 config：本机 GLM-5.3-Flash-Quark-Int8 是
# hidden_size=4096、hc_mult=4 ⇒ hc_attn_fn 形状 (HC*(HC+2), HC*H)=(24,16384)。
# 初版写死 H=5120（另一颗的形状）⇒ residual 与 fn 的第二维对不上；tilelang 与 torch
# 参考拿到的是同一份错形状，比较仍自洽，但那条证据不该靠巧合。此处改为自适应 + 断言。
_cfg=json.load(open(f"{MODEL}/config.json")); _tc=_cfg.get("text_config",_cfg)
HC=int(_tc["hc_mult"]); H=int(_tc["hidden_size"])
fn=ckpt(f"{PFX}layers.0.hc_attn_fn",torch.float32).to(dev)
sc=ckpt(f"{PFX}layers.0.hc_attn_scale",torch.float32).to(dev)
bs=ckpt(f"{PFX}layers.0.hc_attn_base",torch.float32).to(dev)
assert tuple(fn.shape)==(HC*(HC+2), HC*H), f"fn 形状异常 {tuple(fn.shape)} ≠ {(HC*(HC+2), HC*H)}"
print(f"HC={HC} H={H} fn={tuple(fn.shape)} base={tuple(bs.shape)} scale={tuple(sc.shape)}")
torch.manual_seed(0)
residual=(torch.randn(T,HC,H,device=dev)*0.6).to(torch.bfloat16)
pre_mix=torch.softmax(torch.randn(T,HC,device=dev),dim=-1).contiguous()

r0, p0 = residual.clone(), pre_mix.clone()
print(f"调用前 checksum: residual={float(residual.float().abs().sum()):.6f} pre_mix={float(pre_mix.abs().sum()):.6f}")
for i in range(4):
    got = torch.ops.vllm.mhc_pre_delayed_tilelang(residual, fn, sc, bs, RMS, EPS, EPS, 2.0, 20,
                                                  pre_mix, None, None, 1e-6)
    ref = mhc_pre_delayed_torch(r0, fn, sc, bs, RMS, EPS, EPS, 2.0, 20, p0, None)
    e_in  = (got[2].float()-ref[2].float()).abs().max().item()
    e_cmb = (got[1].float()-ref[1].float()).abs().max().item()
    mut_r = float((residual.float()-r0.float()).abs().max())
    mut_p = float((pre_mix.float()-p0.float()).abs().max())
    print(f"  第{i+1}次: layer_input 误差={e_in:.3e} comb误差={e_cmb:.3e} | "
          f"输入是否被改写: residual={mut_r:.3e} pre_mix={mut_p:.3e}")

# 对照腿：回落路径（torch 参考）自身连跑 4 次必须逐位一致 —— 没有这一条，上面那组数只证明了
# 「tilelang 与参考不符」，没证明「不符是 tilelang 的非确定性」而非「参考在抖」。
ref_runs = []
for i in range(4):
    rr = mhc_pre_delayed_torch(r0, fn, sc, bs, RMS, EPS, EPS, 2.0, 20, p0, None)
    ref_runs.append(rr[2].float().clone())
dd = [ (ref_runs[j] - ref_runs[0]).abs().max().item() for j in range(4) ]
print("对照：mhc_pre_delayed_torch 自比 4 次 maxabs = " + " / ".join(f"{x:.3e}" for x in dd))
# tilelang 四次之间的互差（不看参考，只看它自己稳不稳）
tl_runs = []
for i in range(4):
    tl_runs.append(torch.ops.vllm.mhc_pre_delayed_tilelang(residual, fn, sc, bs, RMS, EPS, EPS,
                                                           2.0, 20, pre_mix, None, None, 1e-6)[2].float().clone())
td = [ (tl_runs[j] - tl_runs[0]).abs().max().item() for j in range(4) ]
print("对照：tilelang 自比 4 次 maxabs = " + " / ".join(f"{x:.3e}" for x in td))
ok = max(dd) == 0.0 and max(td) > 1e-3
print("判据: 回落路径确定=%s  tilelang 自身不确定=%s  ⇒ %s" % (
      max(dd) == 0.0, max(td) > 1e-3,
      "根因确认为 tilelang mHC 核的非确定性（gfx90a），回落路径可作正解" if ok else "与本判据不符，需重查"))
print(f"调用后 checksum: residual={float(residual.float().abs().sum()):.6f} pre_mix={float(pre_mix.abs().sum()):.6f}")