#!/usr/bin/env python3
"""INT8 GDN 投影在生产路径（cudagraph replay）里每步到底花多少时间。

背景：eager 调一次 gemm_a8w8_CK 要 ~47 µs（M=6），而访存下限只有 6.55 µs ⇒ 疑似 host 开销。
生产里这些调用在 vLLM 的 cudagraph 内（replay 时无 Python）⇒ 必须用**图内**口径量，否则高估。

本脚本按真实每步模式捕获一张图：45× (M=6,N=2560,K=4096) + 45× (M=6,N=4096,K=1024)
（45 个 GDN 层的 in_proj_qkvz 与 out_proj；用 /metrics 与 aiter 日志确认过这两个 shape）
分别报 eager 与 replay 的每步总时间。
"""
import time

import torch

import aiter  # noqa: F401
from aiter import dtypes
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_CK

M = 6
LAYERS = 45
S1, S2 = (M, 2560, 4096), (M, 4096, 1024)
HBM_GBPS = 1600.0


def mk(m, n, k):
    return (torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8),
            torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8),
            (torch.rand(m, 1, device="cuda") / 90 + 1e-3).float(),
            (torch.rand(n, 1, device="cuda") / 90 + 1e-3).float())


def step_calls(bufs):
    """一次「步」的全部 int8 投影调用（45 层 × 2 个投影）。"""
    for _ in range(LAYERS):
        for (x, w, sx, sw) in bufs:
            gemm_a8w8_CK(x, w, sx, sw, None, dtypes.bf16, 0)


def main():
    b1, b2 = mk(*S1), mk(*S2)
    bufs = [b1, b2]
    floor_ms = sum(LAYERS * (n * k) / (HBM_GBPS * 1e9) * 1e3 for (_, n, k) in (S1, S2))
    print(f"device={torch.cuda.get_device_name(0)}  每步 {2*LAYERS} 次 int8 GEMM")
    print(f"纯访存下限（权重字节/HBM）: {floor_ms:.2f} ms/步\n")

    # ① eager（生产里不这样跑，仅作对照）
    for _ in range(5):
        step_calls(bufs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        step_calls(bufs)
    torch.cuda.synchronize()
    eager = (time.perf_counter() - t0) / 20 * 1e3
    print(f"① eager（含 Python/pybind/host 开销）: {eager:8.2f} ms/步  "
          f"({eager/(2*LAYERS)*1e3:.1f} µs/次)")

    # ② cudagraph capture 后 replay（= vLLM 生产口径）
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step_calls(bufs)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        step_calls(bufs)
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_rep = 200
    for _ in range(n_rep):
        g.replay()
    torch.cuda.synchronize()
    rep = (time.perf_counter() - t0) / n_rep * 1e3
    print(f"② 图内 replay（= 生产路径）        : {rep:8.2f} ms/步  "
          f"({rep/(2*LAYERS)*1e3:.1f} µs/次)")
    print(f"   ⇒ 图内成本 = 步时的 {rep/38.7*100:.2f}%（本臂 step 实测 38.7 ms），"
          f"是访存下限的 {rep/floor_ms:.2f}×")

    # ③ 单次 eager 的 host 开销占比：批量 10 次不同 shape 连发
    print("\n③ 单调用分解（M=6, N=2560, K=4096）：")
    x, w, sx, sw = b1
    for _ in range(20):
        gemm_a8w8_CK(x, w, sx, sw, None, dtypes.bf16, 0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        gemm_a8w8_CK(x, w, sx, sw, None, dtypes.bf16, 0)
    torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / 200 * 1e6
    print(f"   eager 单次 {per:.1f} µs；纯访存下限 {S1[1]*S1[2]/HBM_GBPS/1e9*1e6:.2f} µs"
          f" ⇒ 其中 host/launch 开销 ≈ {per - S1[1]*S1[2]/HBM_GBPS/1e9*1e6:.1f} µs")


if __name__ == "__main__":
    main()
