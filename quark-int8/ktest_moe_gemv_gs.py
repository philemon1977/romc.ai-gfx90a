#!/usr/bin/env python3
"""GEMV MoE 内核（mi250_moe_gemv_gs）数值单测：Triton 实现 vs torch 参考。

背景：该内核原为 Ornith 的 **gs=128** 写的，scale 每 BLOCK_K 只取一个
`k0 // GROUP`；泛化到 DSV4.1 的 **gs=32** 后一个 BLOCK_K=128 跨 4 个 group，
另外 3/4 的 k 乘错 scale（实测输出退化成 "\t **\t **..."）。此测试把两种
group 都钉住，防止回归。

WNA16 布局（compressed-tensors / vLLM int_wna16）：
  B      uint8 [E, N, K//2]   每字节 2 个 int4，**低 nibble = 偶 k**，值 = nibble - 8
  B_scale        [E, N, K//G] 逐 k 元素 scale = B_scale[e, n, k // G]
  A      bf16   [M, K]
  out[p] = topk_weights[p] * sum_k A[p//TOPK, k] * (nib-8) * B_scale[e_p, n, k//G]

用法（容器内，需挂载修好的内核与 sitecustomize）:
  python3 ktest_moe_gemv_gs.py
"""
import os
import sys

import torch

sys.path.insert(0, "/patches/moe_gemv")
os.environ.setdefault("MI250_MOE_GEMV", "1")

import mi250_moe_gemv_gs as G  # noqa: E402


def reference(A, B, S, ids, wts, M, K, N, E, group, topk, apply_w):
    """torch 参考：逐 (token, expert) 对做真正的 GEMV。"""
    # 解包 int4：低 nibble = 偶 k
    lo = (B.to(torch.int16) & 0xF) - 8
    hi = ((B.to(torch.int16) >> 4) & 0xF) - 8
    W = torch.empty((E, N, K), dtype=torch.float32, device=B.device)
    W[:, :, 0::2] = lo.to(torch.float32)
    W[:, :, 1::2] = hi.to(torch.float32)
    # 逐元素 scale
    Sf = S.to(torch.float32)
    scale = torch.repeat_interleave(Sf, group, dim=2)          # [E, N, K]
    assert scale.shape[2] == K, (scale.shape, K)
    out = torch.zeros((M * topk, N), dtype=torch.float32, device=A.device)
    for p in range(M * topk):
        t, e = p // topk, int(ids[p].item())
        e = max(e, 0)
        out[p] = (A[t].to(torch.float32)[None, :] * W[e] * scale[e]).sum(dim=1)
        if apply_w and wts is not None:
            out[p] *= wts[p].to(torch.float32)
    return out


def run_case(group, apply_w, seed=0):
    torch.manual_seed(seed)
    E, N, K = 4, 256, 512
    M, topk = 8, 4
    dev = "cuda"
    A = torch.randn(M, K, dtype=torch.float32, device=dev).to(torch.bfloat16)
    packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
    S = (torch.rand(E, N, K // group, dtype=torch.float32, device=dev) * 0.1 + 0.01)
    ids = torch.randint(0, E, (M * topk,), dtype=torch.int32, device=dev)
    wts = torch.rand(M * topk, dtype=torch.float32, device=dev) if apply_w else None
    C = torch.zeros(M * topk, N, dtype=torch.bfloat16, device=dev)

    G.set_current_topk(ids)
    cfg = {"BLOCK_SIZE_M": 16}
    try:
        took = G.invoke_gemv_wna16(
            A, packed, C, S, None, wts,
            None, None, None,                 # sorted_token_ids/expert_ids/num_tokens
            apply_w, topk, cfg, torch.bfloat16,
            False, True, [1, group],
        )
    finally:
        G.clear_current_topk()
    if not took:
        # apply_w=True 是 gemm2；内核默认只在 MI250_MOE_GEMV_BOTH=1 时接管（收益/开销权衡）
        print(f"  gs={group:3d} apply_w={int(apply_w)}: 内核未接管（skip，属预期）")
        return None
    ref = reference(A, B=packed, S=S, ids=ids, wts=wts, M=M, K=K, N=N, E=E,
                    group=group, topk=topk, apply_w=apply_w)
    got = C[: M * topk].float()
    rel = (got - ref).abs().mean() / (ref.abs().mean() + 1e-9)
    mx = (got - ref).abs().max().item()
    print(f"  gs={group:3d} apply_w={int(apply_w)}: relERR={rel*100:.3f}%  maxAbs={mx:.4f}")
    return rel


def main():
    print("=== GEMV MoE 内核单测（vs torch 参考）")
    bad = 0
    for group in (128, 32):
        for apply_w in (False, True):
            r = run_case(group, apply_w)
            if r is not None and r > 0.01:   # 1% 容差（bf16 输出）
                bad += 1
    print("[PASS] gs=128 与 gs=32 都与参考一致" if bad == 0 else f"[FAIL] {bad} 个用例不符")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
