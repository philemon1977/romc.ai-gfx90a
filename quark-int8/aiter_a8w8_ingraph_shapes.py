#!/usr/bin/env python3
"""生成 gfx90a 的 a8w8 调优行（**us 用图内 replay 口径**），并合成"超集表"。

为什么要图内口径：eager 调一次要 ~47 µs，其中 ~40 µs 是 Python/pybind/host 开销；
生产路径（vLLM cudagraph 内）replay 时没有这部分 ⇒ eager 数字会把成本高估 4 倍。
（实测：90 次/步 eager 4.17 ms vs 图内 1.06 ms。）

本路径只消费 splitK（ops/gemm_op_a8w8.py:660-664），而 gfx90a 上 splitK≥1 直接
RuntimeError("This GEMM is not supported!") ⇒ 唯一合法值 0。其余列均为信息性记录。
"""
import os
import struct
import time

import torch

import aiter  # noqa: F401
from aiter import dtypes
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_CK

VENV_CSV = ("/home/qiba/ai/envs/vllm_0.28.0_rocm72/lib/python3.12/site-packages/"
            "aiter/configs/a8w8_tuned_gemm.csv")
OUT = "/home/qiba/ai/recipes/patches/gfx90a/aiter_a8w8_tuned_gemm_gfx90a.csv"
SHAPES = [(2560, 4096, "in_proj_qkvz"), (4096, 1024, "out_proj")]
MS = [6, 12, 16, 18, 24, 36, 42, 48, 60, 66, 72, 84, 90, 96, 108, 114, 120,
      132, 138, 144, 156, 162, 168, 180, 186, 192, 2048]
HBM_GBPS = 1600.0
REPS = 30


def measure(m, n, k, reps=REPS):
    x = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
    w = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    sx = (torch.rand(m, 1, device="cuda") / 90 + 1e-3).float()
    sw = (torch.rand(n, 1, device="cuda") / 90 + 1e-3).float()

    out = gemm_a8w8_CK(x, w, sx, sw, None, dtypes.bf16, 0)
    torch.cuda.synchronize()
    ref = (x.float() * sx) @ (w.float() * sw).t()
    err = ((out.float() - ref).abs() / (ref.abs() + 1e-3)).max().item()

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(reps):
            gemm_a8w8_CK(x, w, sx, sw, None, dtypes.bf16, 0)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(reps):
            gemm_a8w8_CK(x, w, sx, sw, None, dtypes.bf16, 0)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    nrep = 50
    for _ in range(nrep):
        g.replay()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / nrep / reps * 1e6
    return us, err


def main():
    gfx, cu = get_gfx_runtime(), get_cu_num()
    print(f"gfx={gfx} cu_num={cu} 图内口径，reps={REPS}/图，每 shape ×{len(MS)} 个 M")
    rows = []
    for (n, k, tag) in SHAPES:
        for m in MS:
            us, err = measure(m, n, k)
            floor = n * k / HBM_GBPS / 1e9 * 1e6
            tflops = 2 * m * n * k / (us * 1e-6) / 1e12
            bw = n * k / (us * 1e-6) / 1e9
            rows.append((gfx, cu, m, n, k, "torch.int8", -1, 0, round(us, 4),
                         "ck_a8w8_auto", round(tflops, 4), round(bw, 2), f"{err:.6f}"))
            print(f"  M={m:5d} {tag:14s} 图内 {us:7.2f} µs ({us/floor:5.2f}× 访存下限 "
                  f"{floor:5.2f} µs) bw {bw:6.0f} GB/s err {err:.1e}", flush=True)

    ship = open(VENV_CSV).read().rstrip("\n").split("\n")
    hdr = ship[0]
    assert hdr.split(",")[:6] == ["gfx", "cu_num", "M", "N", "K", "q_dtype_w"], hdr
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(hdr + "\n")
        f.write("\n".join(ship[1:]) + "\n")          # 原表全部行（gfx942/950）原样保留
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    print(f"\n✅ 超集表已写 {OUT}：原表 {len(ship)-1} 行 + gfx90a {len(rows)} 行 "
          f"= {len(ship)-1+len(rows)} 行")

    # 端到端自检：用这张表查一遍（应全部命中，且为 splitK=0）
    import functools
    from aiter.ops.gemm_op_a8w8 import get_GEMM_config_with_quant_type
    get_GEMM_config_with_quant_type.cache_clear()
    hit = miss = 0
    for (n, k, _t) in SHAPES:
        for m in MS:
            cfg = get_GEMM_config_with_quant_type(m, n, k, dtypes.i8, OUT)
            if cfg is None:
                miss += 1
            else:
                hit += 1
                assert cfg["splitK"] == 0, cfg
    print(f"自检：命中 {hit}、未命中 {miss}（全为 splitK=0）")
    # 反向自检：官方表在 gfx90a 上应全部未命中（证明这张表确实是新加的）
    get_GEMM_config_with_quant_type.cache_clear()
    miss_official = sum(get_GEMM_config_with_quant_type(m, n, k, dtypes.i8, VENV_CSV) is None
                        for (n, k, _t) in SHAPES for m in MS)
    print(f"对照：同一批 shape 用官方表 → 未命中 {miss_official}/{len(MS)*len(SHAPES)}（应=54）")


if __name__ == "__main__":
    main()
