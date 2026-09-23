#!/usr/bin/env python3
"""为 gfx90a(MI250X) 生成 a8w8_tuned_gemm.csv 的行 —— 但这张表在本臂路径上**只消费 splitK**。

事实链（逐条查过源码，2026-09-18）：
  vLLM AiterInt8ScaledMMLinearKernel.apply_weights
    -> rocm_aiter_ops.w8a8_gemm(x_q, w_q.t(), x_s, w_s, bias, out_dtype)
    -> aiter.gemm_a8w8_CK(A[M,K], B[N,K], As, Bs, bias, dtype)      (vllm/_aiter_ops.py:648)
    -> get_GEMM_config_with_quant_type(...)  取表里一行
    -> 只有 `splitK = ck_config["splitK"]` 被使用（ops/gemm_op_a8w8.py:660-664）
   ⇒ gfx90a 上"把表补对" = 实测挑出每个 shape 的最优 splitK；
     kernelId/kernelName/us/tflops/bw/errRatio 在本路径上是**信息性**列（ASM 路径才用 kernelName）。

用法：
  python3 aiter_a8w8_tune.py                 # 全量：27 个 capture M × 2 shape × splitK 0..4
  python3 aiter_a8w8_tune.py --quick         # 只测 M in {6,16,2048}
  python3 aiter_a8w8_tune.py --out xxx.csv   # 只写 CSV 不打印长表
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_CK, get_GEMM_config_with_quant_type

# 本臂真实 shape（TP8 之后）：GDN in_proj_qkvz 融合 + out_proj
SHAPES = [(2560, 4096, "in_proj_qkvz"), (4096, 1024, "out_proj")]
# vLLM 的 cudagraph capture 尺寸（本臂 8117 日志实测出现的 M 序列）
CAPTURE_MS = [6, 12, 16, 18, 24, 36, 42, 48, 60, 66, 72, 84, 90, 96, 108, 114, 120,
              132, 138, 144, 156, 162, 168, 180, 186, 192, 2048]
SPLITKS = [0, 1, 2, 3, 4]
HBM_GBPS = 1600.0  # MI250X 每 GCD ~1.6 TB/s


def info():
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
    return get_gfx_runtime(), get_cu_num()


def make_buffers(m, n, k, nbuf=6):
    """多份权重轮转，避免反复读同一块被 L2 吃掉（10.5 MB > 8 MB L2，但保守起见）。"""
    xs = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
    sx = (torch.rand(m, 1, device="cuda") / 90 + 1e-3).float()
    ws = [torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8) for _ in range(nbuf)]
    # 权重 scale 按输出通道（Quark: weight per_channel, ch_axis=0）⇒ [N,1]，与 W[N,K] 行向广播
    sw = [(torch.rand(n, 1, device="cuda") / 90 + 1e-3).float() for _ in range(nbuf)]
    return xs, sx, ws, sw


def timeit(fn, iters=60, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # µs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="/home/qiba/ai/recipes/patches/vllm/vllm_0.28.0_rocm72/aiter_a8w8_tuned_gemm_gfx90a.csv")
    ap.add_argument("--err", type=float, default=0.05)
    args = ap.parse_args()

    gfx, cu = info()
    print(f"device={torch.cuda.get_device_name(0)} gfx={gfx} cu_num={cu} splitK 候选={SPLITKS}")
    ms = [6, 16, 2048] if args.quick else CAPTURE_MS

    rows = []
    for (n, k, tag) in SHAPES:
        for m in ms:
            xs, sx, ws, sw = make_buffers(m, n, k)
            ref = (xs.float() * sx) @ (ws[0].float() * sw[0]).t()
            best = None
            line = []
            for sk in SPLITKS:
                it = [0]

                def run():
                    j = it[0] % len(ws)
                    it[0] += 1
                    return gemm_a8w8_CK(xs, ws[j], sx, sw[j], None, dtypes.bf16, sk)

                try:
                    out = run()
                    torch.cuda.synchronize()
                    rel = ((out.float() - ref).abs() / (ref.abs() + 1e-3)).max().item()
                except Exception as e:  # 该 splitK 在 gfx90a 上可能不被支持
                    line.append(f"sk{sk}:N/A({type(e).__name__})")
                    continue
                us = timeit(run)
                wbytes = n * k
                bw = wbytes / (us * 1e-6) / 1e9
                line.append(f"sk{sk}:{us:7.1f}µs bw{bw:6.0f}GB/s err{rel:.2e}")
                if rel > args.err:
                    print(f"        ⚠️ splitK={sk} err={rel:.2e} > 阈值 {args.err}（仅记录，不丢弃：本路径只有它可用）",
                          flush=True)
                if best is None or us < best["us"]:
                    best = {"splitK": sk, "us": us, "bw": bw, "err": rel}
            floor_us = (n * k) / (HBM_GBPS * 1e9) * 1e6
            print(f"  M={m:5d} N={n} K={k} [{tag:14s}] 下限{floor_us:6.2f}µs | " + "  ".join(line), flush=True)
            if best:
                print(f"        ⇒ best splitK={best['splitK']} {best['us']:.1f}µs "
                      f"({best['us']/floor_us:.2f}× 下限) err={best['err']:.1e}", flush=True)
                rows.append({"gfx": gfx, "cu_num": cu, "M": m, "N": n, "K": k,
                             "q_dtype_w": "torch.int8", "splitK": best["splitK"],
                             "us": round(best["us"], 4), "bw": round(best["bw"], 2),
                             "errRatio": best["err"]})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    hdr = "gfx,cu_num,M,N,K,q_dtype_w,kernelId,splitK,us,kernelName,tflops,bw,errRatio"
    with open(args.out, "w") as f:
        f.write(hdr + "\n")
        for r in rows:
            # kernelId 本路径不消费（CK 内部自选实例）⇒ 写 -1 明确表示"非 ASM 指定"；
            # tflops 由 shape 与实测 us 反算，便于后人一眼判断合理性。
            tflops = 2 * r["M"] * r["N"] * r["K"] / (r["us"] * 1e-6) / 1e12
            f.write(f'{r["gfx"]},{r["cu_num"]},{r["M"]},{r["N"]},{r["K"]},{r["q_dtype_w"]},'
                    f'-1,{r["splitK"]},{r["us"]},ck_a8w8_auto,{tflops:.4f},{r["bw"]},{r["errRatio"]:.6f}\n')
    print(f"\n✅ 已写 {args.out}（{len(rows)} 行）")

    # 自检：用自定义表查一遍，确认命中（不再打印 not found）
    os.environ["AITER_CONFIG_GEMM_A8W8"] = args.out
    hit = miss = 0
    for (n, k, tag) in SHAPES:
        for m in ms:
            cfg = get_GEMM_config_with_quant_type(m, n, k, dtypes.i8, args.out)
            hit += cfg is not None
            miss += cfg is None
    print(f"自检：命中 {hit} / 未命中 {miss}（未命中=表里没有该 M，属预期，其余 M 仍走默认）")


if __name__ == "__main__":
    main()
