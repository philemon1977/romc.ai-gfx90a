#!/usr/bin/env python3
"""GEMV-style int4 W4A16 MoE gemm1 (gate+up) for the tiny-M decode regime, and an A/B
against vLLM's padded fused_moe_kernel_gptq_awq.

Why (measured, see RESULT.md 7sexies): at M=6 tokens/step the incumbent MoE costs
280.8 us/layer (gemm1 213.9) => 16.85 ms/step of a ~37 ms step (46%), while the HBM
bandwidth floor is ~2.0 ms/step (12% of peak). M=1 is 20x off the floor. Root cause:
sorted_ids ~160 => grid = (160/16) x (256/64) = 40 blocks on 104 CUs (under-occupied),
each block serially running 128 K-iterations with BM=16 padding (16x wasted M).

Design here: no sorting, no M padding. One program per (token, expert) pair x N-tile.
  grid = (M*topk, N/BLOCK_N) = (60, 4) = 240 programs, each streaming its expert's
  int4 rows and doing a true GEMV (multiply-accumulate over K, no tl.dot).
Output: per-pair partials [M*topk, N] fp32 (the token-weighted sum across a token's
experts is a separate stage, same as vLLM's moe_sum -- so the comparison is like-for-like).

Layout is exactly vLLM's WNA16 packing: W[e, n, k//8] int32 (nibble j of word i = w[8i+j],
low nibble first), S[e, n, k//group] scales, X[M, K] bf16.
"""
import torch
import triton
import triton.language as tl

GROUP, PACK = 128, 8
PACK_C = tl.constexpr(8)   # Triton 内核内只能访问 constexpr 全局


@triton.jit
def gemv_moe_gate_up_k(
    X, W, S, O, TOPK_IDS, TOPK_W,
    M, K, N, TOPK,
    stride_wk,
    GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_p = tl.program_id(0)                      # (token, expert) pair index
    pid_n = tl.program_id(1)                      # N tile
    t = pid_p // TOPK
    e = tl.load(TOPK_IDS + pid_p)                 # expert of this pair
    wscale = tl.load(TOPK_W + pid_p)              # routing weight of this pair

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k2 = tl.arange(0, BLOCK_K // 2)          # 字节索引（1 字节 = 2 个 int4）

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        # scale: [E, N, K//GROUP]，本 chunk 恰好落在一个 group 内
        sc = tl.load(S + e * N * (K // GROUP) + offs_n * (K // GROUP) + (k0 // GROUP))
        wb = tl.load(W + e * N * (K // 2) + offs_n[:, None] * (K // 2) + (k0 // 2 + offs_k2)[None, :])
        wb = wb.to(tl.int32)
        part = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for j in tl.static_range(2):               # 低 nibble = 偶 k，高 nibble = 奇 k
            nib = (wb >> (4 * j)) & 0xF
            nib = nib - 8                          # vLLM WNA16: 无符号存储 + 隐含零点 8
            xj = tl.load(X + t * K + k0 + (offs_k2 * 2 + j))          # (BK/2,) bf16
            part += tl.sum(nib.to(tl.float32) * xj[None, :].to(tl.float32), axis=1)
        acc += part * sc.to(tl.float32)

    tl.store(O + pid_p * N + offs_n, (acc * wscale).to(O.dtype.element_ty))


def run_ours(x, w_packed, scales, ids_flat, w_flat, N, TOPK, BLOCK_N=64, BLOCK_K=128, warps=4):
    M, K = x.shape
    out = torch.empty(M * TOPK, N, dtype=torch.bfloat16, device=x.device)
    grid = (M * TOPK, N // BLOCK_N)
    gemv_moe_gate_up_k[grid](x, w_packed, scales, out, ids_flat, w_flat,
                             M, K, N, TOPK, w_packed.stride(1),
                             GROUP=GROUP, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                             num_warps=warps)
    return out


def dequant_ref(x, w_bytes, scales, topk_ids, N, K, GROUP_=GROUP):
    """参考：按 vLLM 运行时布局（uint8 字节、每字节 2 个 int4、真值=nibble-8）反量化后再算。"""
    E_ = w_bytes.shape[0]
    wb = w_bytes.to(torch.int64)
    nib = torch.empty(E_, N, K, dtype=torch.float32, device=x.device)
    for j in range(2):
        v = ((wb >> (4 * j)) & 0xF).to(torch.float32) - 8.0
        nib[:, :, j::2] = v
    s = scales.to(torch.float32).repeat_interleave(GROUP_, dim=2)
    wdq = nib * s
    xs = x.to(torch.float32)
    out = torch.empty(x.shape[0] * topk_ids.shape[1], N, dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for kk in range(topk_ids.shape[1]):
            out[t * topk_ids.shape[1] + kk] = wdq[topk_ids[t, kk]] @ xs[t]
    return out


def main():
    torch.manual_seed(0)
    dev = "cuda"
    M, K, E, TOPK = 6, 4096, 512, 10
    N = 256                                        # 2*moe_inter/TP8
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    w_log = torch.randint(-8, 8, (E, N, K), dtype=torch.int64, device=dev)
    w = (((w_log[:, :, 0::2] + 8) & 0xF) | (((w_log[:, :, 1::2] + 8) & 0xF) << 4)).to(torch.uint8).contiguous()
    s = (torch.rand(E, N, K // GROUP, dtype=torch.float32, device=dev) + 0.5).to(torch.bfloat16)
    topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
    topk_w = torch.rand(M, TOPK, dtype=torch.float32, device=dev)
    ids_flat = topk_ids.reshape(-1)
    w_flat = topk_w.reshape(-1)

    print("=== 数值对拍（我的 GEMV vs 反量化参考实现）===")
    ours = run_ours(x, w, s, ids_flat, w_flat, N, TOPK).to(torch.float32)
    ref = dequant_ref(x, w, s, topk_ids, N, K)
    ref_w = ref * w_flat[:, None]
    denom = ref_w.abs().mean().item() + 1e-6
    rel = (ours - ref_w).abs().mean().item() / denom
    print(f"  参考 |均值|={denom:.4f}  我的输出 |均值|={ours.abs().mean().item():.4f}  相对误差={rel*100:.2f}%")

    print("\n=== 计时（同一 harness、同一 shape；现役 gemm1 = 213.9 us/call）===")
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

    for bn, bk, wp in ((64, 128, 4), (128, 128, 4), (64, 64, 4), (256, 128, 8)):
        t = bench(lambda: run_ours(x, w, s, ids_flat, w_flat, N, TOPK, BLOCK_N=bn, BLOCK_K=bk, warps=wp))
        print(f"  ours BLOCK_N={bn:4d} BLOCK_K={bk:4d} warps={wp}: {t:8.1f} us/call   "
              f"(grid={(M*TOPK, N//bn)})  vs 现役 213.9 us  => {(213.9/t-1)*100:+.0f}%")


if __name__ == "__main__":
    main()
