#!/usr/bin/env python3
"""Independent check on already-written shards: per-expert int8 reconstruct vs
the original fused bf16 source tensor (real 4096x1024 expert weights)."""
import json, os, sys
import torch
from safetensors import safe_open

SRC="/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B"
DST="/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-Quark-Int8"
src_idx=json.load(open(os.path.join(SRC,"model.safetensors.index.json")))["weight_map"]
done=sorted(f for f in os.listdir(DST) if f.endswith(".safetensors"))
print("shards written so far:", len(done))
checked=0; errs=[]
for shard in done:
    with safe_open(os.path.join(DST,shard),framework="pt") as f:
        ks=[k for k in f.keys() if k.endswith(".weight") and "experts" in k]
        if not ks: continue
        # one layer per shard is enough
        lay=sorted({k.split("experts.")[0] for k in ks})[0]
        for nm in ("gate_proj","up_proj","down_proj"):
            cand=[k for k in ks if k.startswith(lay) and k.endswith(nm+".weight")]
            if not cand: continue
            k=sorted(cand, key=lambda x:int(x.split("experts.")[1].split(".")[0]))[0]
            e=int(k.split("experts.")[1].split(".")[0])
            t=f.get_tensor(k); s=f.get_tensor(k+"_scale")
            lyr=lay.split("layers.")[1].rstrip(".")
            fused=lay+f"experts."+("gate_up_proj" if nm!="down_proj" else "down_proj")
            with safe_open(os.path.join(SRC,src_idx[fused]),framework="pt") as g:
                full=g.get_tensor(fused)[e]
            orig = full[:full.shape[0]//2] if nm=="gate_proj" else full[full.shape[0]//2:] if nm=="up_proj" else full
            recon=t.float()*s.float().unsqueeze(1)
            errs.append(((recon-orig.float()).abs().max()/(orig.float().abs().max()+1e-9)).item())
            checked+=1
            if checked<=6:
                print(f"  {k}: int8 {tuple(t.shape)} scale {tuple(s.shape)} rel_err {errs[-1]:.4f}")
        if checked>=18: break
errs.sort()
print(f"checked {checked} real expert tensors: median {errs[len(errs)//2]:.4f} p95 {errs[int(len(errs)*0.95)]:.4f} max {errs[-1]:.4f}")
print("PARTIAL_CHECK", "PASS" if errs[-1]<0.05 else "FAIL")
