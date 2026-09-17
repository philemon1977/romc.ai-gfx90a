#!/usr/bin/env python3
"""Validate the vLLM integration wrapper offline before touching the server.

Calls vLLM's upstream WNA16 helper and my invoke_gemv_wna16 with IDENTICAL arguments
(including a real sorted_token_ids/expert_ids layout) and compares outputs + timing.
Timing includes the _expert_of_pair inversion cost, so the reported speedup is the
end-to-end per-layer MoE gain that the server should see.

usage: MI250_MOE_GEMV=1 python test_wrapper.py
"""
import os
import sys

sys.path.insert(0, "/home/qiba/ROCm.AI/quark-int8/moe_gemv_patch")
os.environ.setdefault("MI250_MOE_GEMV", "1")

import torch
import triton.language as tl

from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_wna16_triton_kernel as upstream,
)
from mi250_moe_gemv import invoke_gemv_wna16, GROUP

E, TOPK, HIDDEN, INTER, TP = 512, 10, 4096, 1024, 8
N1, K1 = 2 * (INTER // TP), HIDDEN
N2, K2 = HIDDEN, INTER // TP
CFG = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
       "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": 4, "num_stages": 2}
dev = "cuda"


def mk_weights(N, K):
    wl = torch.randint(-8, 8, (E, N, K), dtype=torch.int64, device=dev)
    b = (((wl[:, :, 0::2] + 8) & 0xF) | (((wl[:, :, 1::2] + 8) & 0xF) << 4)).to(torch.uint8).contiguous()
    s = (torch.rand(E, N, K // GROUP, dtype=torch.float32, device=dev) + 0.5).to(torch.bfloat16)
    return b, s


def build_sorted(pairs_experts, num_valid, block_m):
    """pairs_experts: list of (pair_index, expert) -> vLLM's sorted layout (pad = num_valid)."""
    by_e = {}
    for p, e in pairs_experts:
        by_e.setdefault(e, []).append(p)
    parts, blocks = [], []
    for e, ps in sorted(by_e.items()):
        padded = ((len(ps) + block_m - 1) // block_m) * block_m
        row = torch.full((padded,), num_valid, dtype=torch.int32, device=dev)
        row[:len(ps)] = torch.tensor(ps, dtype=torch.int32, device=dev)
        parts.append(row)
        blocks.append(torch.full((padded // block_m,), e, dtype=torch.int32, device=dev))
    sid = torch.cat(parts) if parts else torch.zeros(block_m, dtype=torch.int32, device=dev)
    eid = torch.cat(blocks) if blocks else torch.zeros(1, dtype=torch.int32, device=dev)
    npp = torch.tensor([sid.numel()], dtype=torch.int32, device=dev)
    return sid, eid, npp


def bench(fn, iters=200, warm=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000.0


def main():
    torch.manual_seed(0)
    M = 6
    print(f"MI250_MOE_GEMV={os.environ.get('MI250_MOE_GEMV')}  M={M} tokens/step")
    for label, (N, K, a_rows, mul_w, top_k) in (
        ("gemm1_gate_up", (N1, K1, M, False, TOPK)),
        ("gemm2_down", (N2, K2, M * TOPK, True, 1)),
    ):
        A = torch.randn(a_rows, K, dtype=torch.bfloat16, device=dev)
        B, S = mk_weights(N, K)
        pairs = a_rows * top_k
        experts = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(a_rows)]).to(torch.int32)
        pairexp = [(p, int(experts[p // top_k, p % top_k])) for p in range(pairs)]
        sid, eid, npp = build_sorted(pairexp, pairs, CFG["BLOCK_SIZE_M"])
        tw = torch.rand(pairs, dtype=torch.float32, device=dev) if mul_w else None
        C_up = torch.zeros(pairs, top_k if not mul_w else 1, N, dtype=torch.bfloat16, device=dev)
        C_me = torch.zeros_like(C_up)
        args = dict(sorted_token_ids=sid, expert_ids=eid,
                    num_tokens_post_padded=npp, mul_routed_weight=mul_w, top_k=top_k,
                    config=CFG, compute_type=tl.bfloat16, use_int8_w8a16=False,
                    use_int4_w4a16=True, block_shape=[0, GROUP])

        tw_arg = None if tw is None else (tw.view(pairs, 1) if mul_w else tw.view(pairs, top_k))

        def f_up():
            upstream(A, B, C_up, S, None, tw_arg, **args)

        took = {"v": None}

        def f_me():
            ok = invoke_gemv_wna16(A, B, C_me, S, None, tw_arg, **args)
            took["v"] = ok
            if not ok:                      # 未接管 => 模拟 vLLM 的回退路径
                upstream(A, B, C_me, S, None, tw_arg, **args)

        f_up(); f_me()
        ref = C_up.reshape(pairs, -1)[:, :N].float()
        mine = C_me.reshape(pairs, -1)[:, :N].float()
        rel = (mine - ref).abs().mean().item() / (ref.abs().mean().item() + 1e-6)
        t_up, t_me = bench(f_up), bench(f_me)
        print(f"\n[{label}] K={K} N={N} pairs={pairs} distinct_experts={len(set(e for _, e in pairexp))} "
              f"takeover={'是' if took['v'] else '否(回退上游)'}")
        print(f"  现役: {t_up:8.1f} us/call   我的(含逆置换): {t_me:8.1f} us/call   => {(t_up/t_me-1)*100:+.0f}%")
        print(f"  数值: 相对误差 {rel*100:.2f}%  {'✅' if rel < 0.05 else '❌'}")
    print("\n（每层节省 = 两个 (现役-我的) 之和；×60 层 = 每步节省 ms）")


if __name__ == "__main__":
    main()
