#!/usr/bin/env python3
"""数值校验：换了 a8w8 kernel 之后，结果是否仍在容差内。

调优器自身用 errRatio（默认 1e-2）判定候选 kernel 是否可接受；我们的启发式改动
同样是换核，所以必须过同一道门。

参考实现：fp32 反量化后做 GEMM ——  out_ref = (XQ * x_scale) @ (WQ * w_scale)^T
被测：aiter.gemm_a8w8(XQ, WQ, x_scale, w_scale, dtype=bf16)

报告 max abs err、以及 errRatio = max|diff| / max|ref|（与调优器同口径的量级）。
"""

import torch

import aiter

SHAPES = [
    (64, 5120, 6144, "o_proj"),
    (64, 5120, 17408, "down"),
    (64, 16384, 5120, "qkv"),
    (64, 34816, 5120, "gate_up"),
    (64, 248320, 5120, "lm_head"),
    (1, 16384, 5120, "qkv_M1"),
    (2, 16384, 5120, "qkv_M2"),
    (128, 16384, 5120, "qkv_M128"),
]

print("%-10s %6s %7s %7s | %12s %12s %10s" % ("shape", "M", "N", "K", "max_abs_err", "max_abs_ref", "ratio"))
print("-" * 78)

torch.manual_seed(0)
worst = 0.0
for (M, N, K, tag) in SHAPES:
    XQ = torch.randint(-100, 100, (M, K), dtype=torch.int8, device="cuda")
    WQ = torch.randint(-100, 100, (N, K), dtype=torch.int8, device="cuda")
    x_scale = torch.rand(M, 1, dtype=torch.float32, device="cuda") * 0.05 + 0.01
    w_scale = torch.rand(N, 1, dtype=torch.float32, device="cuda") * 0.05 + 0.01

    out = aiter.gemm_a8w8(XQ, WQ, x_scale, w_scale, dtype=torch.bfloat16).float()
    ref = (XQ.float() * x_scale) @ (WQ.float() * w_scale).T

    diff = (out - ref).abs()
    denom = ref.abs().max().clamp_min(1e-6)
    ratio = (diff.max() / denom).item()
    worst = max(worst, ratio)
    print("%-10s %6d %7d %7d | %12.4f %12.2f %10.5f" % (
        tag, M, N, K, diff.max().item(), ref.abs().max().item(), ratio))

    del XQ, WQ, x_scale, w_scale, out, ref, diff
    torch.cuda.empty_cache()

print()
print("最差 ratio = %.5f （调优器默认阈值 errRatio = 1e-2）-> %s" % (worst, "在容差内" if worst <= 1e-2 else "超出容差"))
