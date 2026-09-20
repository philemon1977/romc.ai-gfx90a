#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GEMV 变体：把"每 K 步一次跨 lane 归约"改成"二维累加器 + 收尾归约一次"。

依据：moe_gemv_bench.py 实测现有内核 gemm1 = 2.02 ms / 113 MB = 56 GB/s（HBM 3.5%），
而 75 层 × 2.02 ms ≈ 152 ms/token ≈ 6.6 tok/s，与端到端 6.8 tok/s 吻合 ⇒ gemm1 就是主瓶颈。
现写法 `part += tl.sum(nib*xj*sc, axis=1)` 每个 K 步触发一次 [BLOCK_N, BLOCK_K//2] 的跨 lane
归约（48 步 × 2 j = 96 次/输出元素），指令开销主导。本变体：
  acc2[BLOCK_N, BLOCK_K//2] 常驻寄存器，循环内只做 FMA；收尾 tl.sum(acc2, axis=1) 一次。
"""
import triton
import triton.language as tl


@triton.jit
def _gemv_moe_acc2_kernel(X, W, S, O, IDS, WTS, M, K, N, TOPK,
                          APPLY_W: tl.constexpr, GROUP: tl.constexpr,
                          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_p = tl.program_id(0)
    pid_n = tl.program_id(1)
    t = pid_p // TOPK
    e = tl.maximum(tl.load(IDS + pid_p), 0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k2 = tl.arange(0, BLOCK_K // 2)
    acc2 = tl.zeros((BLOCK_N, BLOCK_K // 2), dtype=tl.float32)   # 2D 累加器（关键改动）
    for k0 in range(0, K, BLOCK_K):
        wb = tl.load(W + e * N * (K // 2) + offs_n[:, None] * (K // 2) + (k0 // 2 + offs_k2)[None, :])
        wb = wb.to(tl.int32)
        for j in tl.static_range(2):
            kk = k0 + offs_k2 * 2 + j
            sc = tl.load(S + e * N * (K // GROUP) + offs_n[:, None] * (K // GROUP) + (kk // GROUP)[None, :])
            nib = ((wb >> (4 * j)) & 0xF) - 8
            xj = tl.load(X + t * K + kk)
            acc2 += nib.to(tl.float32) * xj[None, :].to(tl.float32) * sc.to(tl.float32)
    acc = tl.sum(acc2, axis=1)                                   # 只归约一次
    if APPLY_W:
        acc = acc * tl.load(WTS + pid_p)
    tl.store(O + pid_p * N + offs_n, acc.to(O.dtype.element_ty))
