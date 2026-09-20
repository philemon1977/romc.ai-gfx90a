# SPDX-License-Identifier: Apache-2.0
"""v3：把 group scale 从「逐元素 gather」提到「每 group 一次」。

定位实验（quark-int8/moe_gemv_diag.py，单 GCD MI250X，真实 checkpoint 布局）：
    全量 v2            1674.5 us   60.1 GB/s
    去 scale（不乘）    378.2 us  266.2 GB/s   ← 4.4x
    去 nibble 提取     1663.9 us   60.5 GB/s   ← nibble 几乎免费
    纯 load 同访存模式  112-162 us 620-898 GB/s
根因：v1/v2 里 scale 的下标是 kk // GROUP（kk = k0 + 2c + j 逐元素），Triton 无法
向量化 ⇒ 每个 k 步退化成 [BLOCK_N, BLOCK_K//2] 次 2 字节 gather（BLOCK_K=64 时 2048
次取指，其中只有 2 个不同地址）≈16x 指令放大。

v3 结构：BLOCK_K = G_PER_STEP * GROUP，三维累加器 [BLOCK_N, G_PER_STEP, GROUP//2]，
循环内完全不碰 scale；每个 k 步结束时把第三维归约掉、只乘一次 [BLOCK_N, G_PER_STEP]
的 scale 切片（同一行内这 G 个值在内存里连续）。
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _gemv_moe_v3(X, W, S, O, IDS, WTS, M, K, N, TOPK,
                 APPLY_W: tl.constexpr, GROUP: tl.constexpr, G_PER_STEP: tl.constexpr,
                 BLOCK_N: tl.constexpr):
    """一程序一 (token, expert) 对 x N 片；BLOCK_K = G_PER_STEP * GROUP。"""
    pid_p = tl.program_id(0)
    pid_n = tl.program_id(1)
    t = pid_p // TOPK
    e = tl.maximum(tl.load(IDS + pid_p), 0)
    HALF: tl.constexpr = GROUP // 2                 # 每 group 的字节数
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_g = tl.arange(0, G_PER_STEP)
    offs_h = tl.arange(0, HALF)
    row_w = offs_n[:, None, None] * (K // 2)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, G_PER_STEP * GROUP):
        acc3 = tl.zeros((BLOCK_N, G_PER_STEP, HALF), dtype=tl.float32)
        for j in tl.static_range(2):
            wo = (k0 // 2 + offs_g[None, :, None] * HALF + offs_h[None, None, :])
            wb = tl.load(W + e * N * (K // 2) + row_w + wo)                     # [BN, G, HALF] uint8
            xo = (k0 + offs_g[:, None] * GROUP + offs_h[None, :] * 2 + j)       # [G, HALF]
            xj = tl.load(X + t * K + xo)
            acc3 += (((wb.to(tl.int32) >> (4 * j)) & 0xF) - 8).to(tl.float32) * xj[None, :, :].to(tl.float32)
        # 每步只取一次 scale：[BN, G]，同一行内这 G 个值连续
        sg = tl.load(S + e * N * (K // GROUP) + offs_n[:, None] * (K // GROUP)
                     + (k0 // GROUP + offs_g)[None, :])
        acc += tl.sum(tl.sum(acc3, axis=2) * sg.to(tl.float32), axis=1)
    if APPLY_W:
        acc = acc * tl.load(WTS + pid_p)
    tl.store(O + pid_p * N + offs_n, acc.to(O.dtype.element_ty))
