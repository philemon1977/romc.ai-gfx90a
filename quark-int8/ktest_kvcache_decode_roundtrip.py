#!/usr/bin/env python3
"""KV cache 写→读 往返（C++ 编码器 vs Triton decode 读端）

写一行已知 KV（恒等 RoPE）→ 让 rocm_sparse_attn_decode 只选这一行 → 输出必须等于该行。
已验证：gfx90a 上写端与读端 fp8_ds_mla 布局/scale 约定一致（nope=0.5/rope=0.25 逐段复现）。

用法（容器内，需 GPU）: python3 ktest_kvcache_decode_roundtrip.py
"""
import torch
sys.path.insert(0,"/patches")
HEAD_DIM=512; ROPE_DIM=64; NOPE_DIM=HEAD_DIM-ROPE_DIM; ROW=HEAD_DIM+4; BS=64
import vllm
from vllm.platforms import current_platform
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_decode
dev="cuda"
cos_sin=torch.zeros(64,ROPE_DIM,dtype=torch.float32,device=dev)
cos_sin[:,:ROPE_DIM//2]=1.0
# 已知输入：nope = 0.5 常量, rope = 0.25 常量（恒等 RoPE ⇒ 写进缓存的就是它）
kv=torch.zeros(1,HEAD_DIM,dtype=torch.bfloat16,device=dev)
kv[:,:NOPE_DIM]=0.5; kv[:,NOPE_DIM:]=0.25
cache3=torch.zeros(2,BS,ROW,dtype=torch.uint8,device=dev); cache=cache3.view(2,BS*ROW)
slots=torch.tensor([0],dtype=torch.int64,device=dev)
pos=torch.tensor([0],dtype=torch.int64,device=dev)
q=torch.zeros(1,8,HEAD_DIM,dtype=torch.bfloat16,device=dev); q[:,:, :NOPE_DIM]=0.01
torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(q,kv,cache,slots,pos,cos_sin,8,1e-6,BS,False)
torch.cuda.synchronize()
print("缓存已写入；输入 kv nope=0.5 rope=0.25")
# 读端：单 query、只选 slot 0 一行、无 sink
swa_idx=torch.full((1,1,128),-1,dtype=torch.int32,device=dev); swa_idx[0,0,0]=0
swa_lens=torch.ones(1,dtype=torch.int32,device=dev)
out=torch.zeros(1,8,HEAD_DIM,dtype=torch.bfloat16,device=dev)
q2=torch.zeros(1,8,HEAD_DIM,dtype=torch.bfloat16,device=dev); q2[:,:, :NOPE_DIM]=0.01
rocm_sparse_attn_decode(
    q=q2, kv_cache=None, swa_k_cache=cache3, swa_only=True,
    topk_indices=None, topk_lens=None,
    swa_indices=swa_idx, swa_lens=swa_lens,
    swa_ragged_indices=None, swa_ragged_indptr=None,
    topk_ragged_indices=None, topk_ragged_indptr=None,
    attn_sink=None, scale=1.0, head_dim=HEAD_DIM,
    nope_head_dim=NOPE_DIM, rope_head_dim=ROPE_DIM, output=out,
)
torch.cuda.synchronize()
o=out[0,0].float()
print(f"读端输出 out[0,0,:4]={[round(float(x),4) for x in o[:4]]}  "
      f"out[..,444:452]={[round(float(x),4) for x in o[444:452]]}  "
      f"out[..,508:512]={[round(float(x),4) for x in o[508:512]]}")
print(f"输出统计: absmax={o.abs().max().item():.5g} mean={o.mean().item():.5g} "
      f"nope段均值={o[:NOPE_DIM].mean().item():.5g} rope段均值={o[NOPE_DIM:].mean().item():.5g}")
print("期望：单行选中且无 sink ⇒ 输出应≈缓存行的值（0.5 / 0.25 量级）")
