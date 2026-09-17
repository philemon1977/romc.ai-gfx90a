#!/usr/bin/env python3
"""用捕获的真实张量给 gemm1 重新定价：真实路由（10 个不同专家）下 现役 vs 我的。"""
import torch, triton.language as tl, sys
sys.path.insert(0, "/home/qiba/ROCm.AI/quark-int8")
from gemv_moe import run_ours
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_wna16_triton_kernel as up

d = torch.load("/tmp/mi250_moe_real.pt", map_location="cuda")
A, B, S = d["A"].cuda(), d["B"].cuda(), d["B_scale"].cuda()
ids = d["topk_ids"].cuda(); N, K, top_k = d["N"], d["K"], d["top_k"]
M = A.shape[0]; pairs = M * top_k; cfg = d["config"]
import os
REP = int(os.environ.get("REP", "1"))
if REP > 1:                       # 用复制把形状放大到真实 decode 规模（M=24 → 240 对）
    A = A.repeat(REP, 1); ids = ids.repeat(REP, 1)
    M = A.shape[0]; pairs = M * top_k
sid, eid, npp = d["sorted"].cuda(), d["expert_ids"].cuda(), torch.tensor([d["num_valid"]], dtype=torch.int32, device="cuda")
C = torch.zeros(M, top_k, N, dtype=torch.bfloat16, device="cuda")
ids_flat = ids.reshape(-1)[:pairs].contiguous()

def bench(fn, it=300, w=50):
    for _ in range(w): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1000

t_up = bench(lambda: up(A, B, C, S, None, None, sid, eid, npp, False, top_k, cfg, tl.bfloat16, False, True, [0,128]))
t_me = bench(lambda: run_ours(A, B, S, ids_flat, torch.ones(pairs, device="cuda"), N, top_k, BLOCK_N=128, BLOCK_K=128, warps=4))
print(f"真实路由（A 取捕获的 {M} 行 → {pairs} 对、{len(torch.unique(ids_flat).tolist())} 个不同专家）")
print(f"  现役 gemm1: {t_up:7.1f} us/call")
print(f"  我的 gemm1: {t_me:7.1f} us/call   => {(t_up/t_me-1)*100:+.0f}%")
print(f"  每层省 {(t_up-t_me)/1000:.3f} ms  ⇒ ×60 层 = {(t_up-t_me)*60/1000:.2f} ms/步")
