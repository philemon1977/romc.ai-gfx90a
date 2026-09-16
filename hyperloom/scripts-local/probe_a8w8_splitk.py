#!/usr/bin/env python3
"""判定实验：M=64 的 decode GEMM 能否靠 splitK 变快。

背景：本机 AITER 的 a8w8 调优库（configs/a8w8_tuned_gemm.csv）只有 gfx942 行、cu_num∈{80,256}，
而本机 get_gfx()='gfx90a'、get_cu_num()=104 → 454 次 "not found tuned config, will use default config"。
未命中时 `gemm_a8w8_CK` 走 `splitK = 0`（无 split-K）。

形取自真实 server 日志（M 为 decode 批量）：
    [M,5120]x[16384,5120]     qkv
    [M,5120]x[5120,6144]      o_proj
    [M,5120]x[34816,5120]     gate/up
    [M,5120]x[1024,17408]     down   (N=5120, K=17408)
    [M,5120]x[14336,5120]
    [M,5120]x[248320,5120]    lm_head

对每个 (M,N,K) 扫 splitK ∈ {0,1,2,4,8}，报告 ms 与有效权重读带宽 GB/s，
并与"纯读上限 1372 GB/s"和"M=1 gemv 实测 721 GB/s"对照。

用法（容器内）：ROCR_VISIBLE_DEVICES=0 python3 probe_a8w8_splitk.py [--m 64]
"""

import argparse
import sys
import time

import torch

import aiter

SHAPES = [  # (N, K) —— M 由命令行给定
    (16384, 5120, "qkv"),
    (5120, 6144, "o_proj"),
    (34816, 5120, "gate_up"),
    (5120, 17408, "down"),
    (14336, 5120, "mlp_mid"),
    (248320, 5120, "lm_head"),
]

PURE_READ_GBPS = 1372.0   # 探针 A 的 sum
GEMV_GBPS = 721.0         # 探针 A 的 gemv (M=1)


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--splitk", type=str, default="0,1,2,4,8")
    args = ap.parse_args()
    M = args.m
    sks = [int(x) for x in args.splitk.split(",")]

    print("torch %s | aiter %s | M=%d" % (torch.__version__, getattr(aiter, "__version__", "?"), M))
    print("参考：纯读上限 %.0f GB/s ；M=1 gemv 实测 %.0f GB/s" % (PURE_READ_GBPS, GEMV_GBPS))
    hdr = "%-10s %7s %7s | %s" % ("shape", "N", "K", "  ".join("sk=%d" % s for s in sks))
    print(hdr)
    print("-" * len(hdr))

    for (N, K, tag) in SHAPES:
        XQ = torch.randint(-127, 127, (M, K), dtype=torch.int8, device="cuda")
        WQ = torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda")
        x_scale = torch.rand(M, 1, dtype=torch.float32, device="cuda") + 0.5
        w_scale = torch.rand(N, 1, dtype=torch.float32, device="cuda") + 0.5

        wbytes = N * K  # int8
        cells = []
        for sk in sks:
            def run(sk=sk):
                return aiter.gemm_a8w8(XQ, WQ, x_scale, w_scale, dtype=torch.bfloat16, splitK=sk)
            try:
                dt = bench(run)
                gbps = wbytes / dt / 1e9
                cells.append("%6.2fms/%4.0f" % (dt * 1e3, gbps))
            except Exception as e:
                cells.append("  ERR(%s)" % type(e).__name__)
        print("%-10s %7d %7d | %s" % (tag, N, K, "  ".join(cells)))

        del XQ, WQ, x_scale, w_scale
        torch.cuda.empty_cache()

    print()
    print("每格 = 毫秒 / 有效权重读带宽 GB/s")


if __name__ == "__main__":
    sys.exit(main())
