#!/usr/bin/env python3
"""Microbenchmark vLLM's WNA16 Triton MoE kernel at THIS model's shapes.

Why: no profiler is usable here (torch --profiler-config and ROCPROF_KERNEL_TRACE both kill
EngineCore), so the MoE's real cost was never measured. This calls vLLM's own
invoke_fused_moe_wna16_triton_kernel with synthetic tensors in the exact layout, one GCD,
per-rank TP8 sharding, and extrapolates to a full step (60 layers x 2 gemms).

Per-rank shapes (TP8), E=512, topk=10, hidden=4096, moe_inter=1024, group_size=128:
  gemm1 (gate+up): A[M,4096] x B[E, 256, 4096] -> C[M,topk,256]     (256 = 2*1024/8)
  gemm2 (down)   : A[M*topk,128] x B[E, 4096, 128] -> C[M,4096]     (128 = 1024/8)
Weight bytes per rank per expert = 0.52 + 0.26 MB; ~57 distinct experts/step -> ~44 MB/layer
-> ~2.7 GB/step/rank -> ~2.1 ms/step at 1.25 TB/s (the GEMV bandwidth floor).

The step is ~37 ms (89.78 t/s at 3.34 tokens/step), so the measured MoE time here tells us
the MoE's true share and therefore the prize for a decode-shaped (GEMV) MoE kernel.
"""
import torch
import triton.language as tl

from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_wna16_triton_kernel,
)
from vllm.utils.torch_utils import set_random_seed

E, TOPK, HIDDEN, INTER = 512, 10, 4096, 1024
GROUP, PACK, TP = 128, 8, 8
INTER_R = INTER // TP          # 128
N1 = 2 * INTER_R               # 256  (gate+up per rank)
N2 = HIDDEN                    # 4096 (down output, unsharded)
K2 = INTER_R                   # 128  (down input per rank)

CFG = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
       "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": 4, "num_stages": 2}


def routing(M, topk, E, block_m, device, pad_value=None):
    """Build sorted_token_ids / expert_ids / num_tokens_post_padded like vLLM does."""
    ids = torch.stack([torch.randperm(E, device=device)[:topk] for _ in range(M)])
    pv = M * topk if pad_value is None else pad_value
    parts, experts = [], []
    for e in range(E):
        toks = (ids == e).any(dim=1).nonzero(as_tuple=True)[0]
        if toks.numel() == 0:
            continue
        n = toks.numel()
        padded = ((n + block_m - 1) // block_m) * block_m
        row = torch.full((padded,), pv, dtype=torch.int32, device=device)  # pad -> out of range
        row[:n] = toks.to(torch.int32)
        parts.append(row)
        experts.append(torch.full((padded // block_m,), e, dtype=torch.int32, device=device))
    sorted_ids = torch.cat(parts) if parts else torch.zeros(block_m, dtype=torch.int32, device=device)
    expert_ids = torch.cat(experts) if experts else torch.zeros(1, dtype=torch.int32, device=device)
    num_post = torch.tensor([sorted_ids.numel()], dtype=torch.int32, device=device)
    return ids.to(torch.int32), sorted_ids, expert_ids, num_post


def bench(fn, iters=100, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0     # us


def main():
    set_random_seed(0)
    dev = "cuda"
    torch.cuda.set_device(0)
    print(f"device={torch.cuda.get_device_name(0)}  E={E} topk={TOPK} hidden={HIDDEN} inter={INTER} TP={TP}")

    for M in (1, 6):
        # ---- gemm1: A[M,HIDDEN] x B[E,N1,K=HIDDEN] ----
        a1 = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device=dev)
        wl = torch.randint(-8, 8, (E, N1, HIDDEN), dtype=torch.int64, device=dev)
        b1 = (((wl[:, :, 0::2] + 8) & 0xF) | (((wl[:, :, 1::2] + 8) & 0xF) << 4)).to(torch.uint8).contiguous()
        s1 = torch.rand(E, N1, HIDDEN // GROUP, dtype=torch.bfloat16, device=dev) + 0.5
        c1 = torch.zeros(M, TOPK, N1, dtype=torch.bfloat16, device=dev)
        ids1, sid1, eid1, npp1 = routing(M, TOPK, E, CFG["BLOCK_SIZE_M"], dev)
        tw1 = torch.rand(M, TOPK, dtype=torch.float32, device=dev)

        def f1():
            invoke_fused_moe_wna16_triton_kernel(
                a1, b1, c1, s1, None, tw1, sid1, eid1, npp1,
                False, TOPK, CFG, tl.bfloat16, False, True, [0, GROUP])

        t1 = bench(f1)

        # ---- gemm2: A[M*topk,K2] x B[E,N2,K2] ----
        a2 = torch.randn(M * TOPK, K2, dtype=torch.bfloat16, device=dev)
        wl2 = torch.randint(-8, 8, (E, N2, K2), dtype=torch.int64, device=dev)
        b2 = (((wl2[:, :, 0::2] + 8) & 0xF) | (((wl2[:, :, 1::2] + 8) & 0xF) << 4)).to(torch.uint8).contiguous()
        s2 = torch.rand(E, N2, K2 // GROUP, dtype=torch.bfloat16, device=dev) + 0.5
        c2 = torch.zeros(M * TOPK, 1, N2, dtype=torch.bfloat16, device=dev)
        # gemm2: 每个 A 行就是一个 (token,expert) 对 => 与 vLLM 一致地用 top_k=1，路由按行号分组
        ids2, sid2, eid2, npp2 = routing(M * TOPK, 1, E, CFG["BLOCK_SIZE_M"], dev, pad_value=M * TOPK)
        tw2 = torch.rand(M * TOPK, 1, dtype=torch.float32, device=dev)

        def f2():
            invoke_fused_moe_wna16_triton_kernel(
                a2, b2, c2, s2, None, tw2, sid2, eid2, npp2,
                True, 1, CFG, tl.bfloat16, False, True, [0, GROUP])

        t2 = bench(f2)

        per_layer = t1 + t2
        per_step = per_layer * 60 / 1000.0
        print(f"\nM={M} tokens/step   shape(gemm1) A[{M},{HIDDEN}]xB[{E},{N1},{HIDDEN//2}]u8")
        print(f"  gemm1 (gate+up) : {t1:8.1f} us/call")
        print(f"  gemm2 (down)    : {t2:8.1f} us/call")
        print(f"  per layer       : {per_layer:8.1f} us   (sorted_ids gemm1={sid1.numel()}, gemm2={sid2.numel()})")
        print(f"  x60 layers      : {per_step:8.2f} ms/step  <-- compare with the measured ~37 ms/step")

    print("\n注：权重字节/rank/expert = %.2f MB；~57 活跃专家 -> %.0f MB/layer -> %.1f GB/step -> 带宽下界 ≈ %.1f ms/step"
          % ((N1 * HIDDEN + N2 * K2) * 0.5 / 2**20,
             (N1 * HIDDEN + N2 * K2) * 0.5 / 2**20 * 57,
             (N1 * HIDDEN + N2 * K2) * 0.5 / 2**30 * 57 * 60,
             (N1 * HIDDEN + N2 * K2) * 0.5 / 2**30 * 57 * 60 / 1250))


if __name__ == "__main__":
    main()
