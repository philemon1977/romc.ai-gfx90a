#!/usr/bin/env python3
"""最后一个廉价候选：bpreshuffle 路径在 M=64 下是否比普通 CK a8w8 快。

背景（见 probe_a8w8_splitk.py 的结果）：
  M=64 下普通 CK a8w8 的有效权重读带宽随 N 单调上升（222→554 GB/s），
  说明是 grid/占用率受限（BlockN=128，N=5120 → 仅 40 个 CTA，本机 104 CU）。
  普通路径的 splitK>0 在本机抛 RuntimeError（预编译模块无 splitK 变体）。

bpreshuffle 把权重预重排成对 GEMM 更友好的布局，是 AITER 对 a8w8 推理推荐的快路径，
且本机已预编译 module_gemm_a8w8_bpreshuffle.so。

用法：ROCR_VISIBLE_DEVICES=0 python3 probe_a8w8_bpreshuffle.py [--m 64]
"""

import argparse
import time

import torch

import aiter
from aiter.ops.shuffle import shuffle_weight

SHAPES = [
    (5120, 6144, "o_proj"),
    (14336, 5120, "mlp_mid"),
    (16384, 5120, "qkv"),
    (34816, 5120, "gate_up"),
    (5120, 17408, "down"),
    (248320, 5120, "lm_head"),
]


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
    ap.add_argument("--layout", type=int, default=16, help="shuffle layout, e.g. 16 -> (16,16)")
    args = ap.parse_args()
    M = args.m
    lay = (args.layout, args.layout)

    print("M=%d | shuffle layout=%s" % (M, lay))
    hdr = "%-9s %7s %7s | %13s %13s %8s" % ("shape", "N", "K", "plain ms/GBps", "bpre ms/GBps", "speedup")
    print(hdr)
    print("-" * len(hdr))

    for (N, K, tag) in SHAPES:
        XQ = torch.randint(-127, 127, (M, K), dtype=torch.int8, device="cuda")
        WQ = torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda")
        x_scale = torch.rand(M, 1, dtype=torch.float32, device="cuda") + 0.5
        w_scale = torch.rand(N, 1, dtype=torch.float32, device="cuda") + 0.5
        wbytes = N * K

        def plain():
            return aiter.gemm_a8w8(XQ, WQ, x_scale, w_scale, dtype=torch.bfloat16)

        try:
            d0 = bench(plain)
            s0 = "%7.2f/%5.0f" % (d0 * 1e3, wbytes / d0 / 1e9)
        except Exception as e:
            d0, s0 = None, "ERR(%s)" % type(e).__name__

        try:
            WQs = shuffle_weight(WQ, layout=lay)
            Out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

            def bpre():
                return aiter.gemm_a8w8_bpreshuffle_ck(XQ, WQs, x_scale, w_scale, Out, 0)

            d1 = bench(bpre)
            s1 = "%7.2f/%5.0f" % (d1 * 1e3, wbytes / d1 / 1e9)
            sp = "%7.2fx" % (d0 / d1) if d0 else "   n/a"
        except Exception as e:
            s1, sp = "ERR(%s)" % str(e)[:24], "   n/a"

        print("%-9s %7d %7d | %13s %13s %8s" % (tag, N, K, s0, s1, sp))

        del XQ, WQ, x_scale, w_scale
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
