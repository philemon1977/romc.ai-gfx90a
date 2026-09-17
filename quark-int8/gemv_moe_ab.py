#!/usr/bin/env python3
"""Full A/B: my GEMV-style decode MoE vs vLLM's padded WNA16 Triton MoE, at this model's
exact shapes, in the TRUE runtime layout.

Runtime layout (pinned from vllm/.../fused_moe/oracle/int_wna16.py:1630, compressed-tensors
path): B arrives as uint8 [E, N, K//2] (each byte = 2 int4, low nibble = even k, value =
nibble - 8) and scales as [E, N, K//group]; A is [M, K] bf16; gemm2 uses top_k=1 with
A = [M*topk, K2] and C = [M*topk, 1, N2].

Per-rank (TP8) shapes: gemm1 N=256 K=4096 ; gemm2 N=4096 K=128 ; E=512 topk=10.
"""
import torch
import triton.language as tl

from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_wna16_triton_kernel,
)
from gemv_moe import run_ours

E, TOPK, HIDDEN, INTER, GROUP = 512, 10, 4096, 1024, 128
TP = 8
N1, K1 = 2 * (INTER // TP), HIDDEN       # gate+up : 256 x 4096
N2, K2 = HIDDEN, INTER // TP             # down    : 4096 x 128
CFG = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
       "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": 4, "num_stages": 2}
dev = "cuda"


def mk_weights(E_, N, K):
    w_log = torch.randint(-8, 8, (E_, N, K), dtype=torch.int64, device=dev)
    b = (((w_log[:, :, 0::2] + 8) & 0xF) | (((w_log[:, :, 1::2] + 8) & 0xF) << 4)).to(torch.uint8).contiguous()
    s = (torch.rand(E_, N, K // GROUP, dtype=torch.float32, device=dev) + 0.5).to(torch.bfloat16)
    return b, s


def routing(M, block_m, pad_value):
    """sorted_token_ids / expert_ids / num_tokens_post_padded, one expert per token (TOPK set below)."""
    ids = torch.arange(TOPK, device=dev).repeat(M)          # token i -> experts 0..TOPK-1
    parts, experts = [], []
    for e in range(TOPK):
        row = torch.full((block_m,), pad_value, dtype=torch.int32, device=dev)
        row[:1] = (e * M) if False else 0                    # placeholder, replaced below
        parts.append(row); experts.append(torch.tensor([e], dtype=torch.int32, device=dev))
    return ids.to(torch.int32), parts, experts


def build_routing(rows, block_m, pad_value):
    """rows: list of (expert, row_index)"""
    parts, experts = [], []
    by_e = {}
    for r, e in rows:
        by_e.setdefault(e, []).append(r)
    for e, rs in sorted(by_e.items()):
        n = len(rs)
        padded = ((n + block_m - 1) // block_m) * block_m
        row = torch.full((padded,), pad_value, dtype=torch.int32, device=dev)
        row[:n] = torch.tensor(rs, dtype=torch.int32, device=dev)
        parts.append(row); experts.append(torch.full((padded // block_m,), e, dtype=torch.int32, device=dev))
    return torch.cat(parts), torch.cat(experts), torch.tensor([sum(p.numel() for p in parts)], dtype=torch.int32, device=dev)


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
    for gemm, (N, K, A_rows, mul_w) in (("gemm1_gate_up", (N1, K1, M, False)),
                                        ("gemm2_down", (N2, K2, M * TOPK, True))):
        x = torch.randn(A_rows, K, dtype=torch.bfloat16, device=dev)
        b, s = mk_weights(E, N, K)
        top_k = 1 if mul_w else TOPK
        wts = torch.ones(A_rows * top_k, dtype=torch.float32, device=dev)
        if not mul_w:
            # gemm1: 每个 token 随机取 topk 个不同专家（贴近真实路由）
            tk = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(A_rows)]).to(torch.int32)
            ids = tk.reshape(-1)
            rows = [(t * TOPK + j, int(tk[t, j])) for t in range(A_rows) for j in range(TOPK)]
        else:
            tk = torch.stack([torch.randperm(E, device=dev)[:1] for _ in range(A_rows)]).to(torch.int32)
            ids = tk.reshape(-1)
            rows = [(r, int(tk[r, 0])) for r in range(A_rows)]
        sid, eid, npp = build_routing(rows, CFG["BLOCK_SIZE_M"], A_rows)
        c = torch.zeros(A_rows, top_k, N, dtype=torch.bfloat16, device=dev) if not mul_w else torch.zeros(A_rows, 1, N, dtype=torch.bfloat16, device=dev)

        def f_inc():
            invoke_fused_moe_wna16_triton_kernel(
                x, b, c, s, None, wts.view(A_rows, top_k), sid, eid, npp,
                mul_w, top_k, CFG, tl.bfloat16, False, True, [0, GROUP])

        t_inc = bench(f_inc)
        # 我的内核：BN=128 最优；输出 [A_rows, N]
        for bn, bk, wp in ((128, 128, 4), (64, 256, 8), (256, 64, 8)):
            try:
                t_mine = bench(lambda: run_ours(x, b, s, ids, wts, N, TOPK if not mul_w else 1,
                                                BLOCK_N=bn, BLOCK_K=bk, warps=wp))
            except Exception as ex:
                print(f"  ours BN={bn} BK={bk} w={wp}: 失败 {type(ex).__name__}"); continue
            print(f"  我的 BN={bn:4d} BK={bk:4d} w={wp}: {t_mine:8.1f} us/call  => {(t_inc/t_mine-1)*100:+.0f}% vs 现役  "
                  f"(省 {(t_inc-t_mine)/1000:.3f} ms/layer)")
        t_mine = None
        ref = c.reshape(A_rows, -1)[:, :N].float() if not mul_w else c.reshape(A_rows, -1).float()
        print(f"\n[{gemm}]  A[{A_rows},{K}] x B[E={E},{N},{K//2}]u8   K={K} N={N}  "
              f"distinct_experts={len(set(e for _, e in rows))}")
        print(f"  现役 : {t_inc:8.1f} us/call")
        mo = run_ours(x, b, s, ids, wts, N, TOPK if not mul_w else 1, BLOCK_N=128, BLOCK_K=128, warps=4).float()
        ref2 = c.reshape(-1, N).float()
        rr = (mo - ref2).abs().mean().item() / (ref2.abs().mean().item() + 1e-6)
        print(f"  数值对拍（我的 vs 现役）: 相对误差 {rr*100:.2f}%  {'✅ 一致' if rr < 0.05 else '❌ 不一致'}")

    print("\n=== 外推（60 层/步）===")
    print("  （把两次打印的 (现役-我的) 相加 ×60 即每步节省；步长基准 ~37 ms @ 89.78 t/s）")


if __name__ == "__main__":
    main()
