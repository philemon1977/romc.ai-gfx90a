#!/usr/bin/env python3
"""Measure what the INT8 GEMM paths actually achieve on gfx90a (MI250X).

Compares:
  1. aiter CK  (gemm_a8w8)      -- AMD's Composable Kernel int8 path, built by JIT for gfx90a
  2. torch._int_mm / hipBLASLt  -- the vendor library path
and reports TOPS vs the CDNA2 theoretical int8 peak per GCD (~191 TOPS for MI250X
at 1.7 GHz: 2x 383 TFLOPS package peak / 2 GCDs).

Also runs a Triton int8 matmul when available, to see whether Triton's AMD backend
lowers int8 dot to MFMA int8 or falls back to FMA (which shows up as a very low
TOPS number).
"""
import time

import torch

PEAK_TOPS = 191.0  # per GCD, dense int8, CDNA2


def tops(m, n, k, dt):
    return 2 * m * n * k / dt / 1e12


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main() -> None:
    print(f"device: {torch.cuda.get_device_name(0)}")
    for (m, n, k) in ((1, 4096, 4096), (16, 4096, 4096), (256, 4096, 4096), (2048, 4096, 4096)):
        x8 = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
        w8 = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
        sx = torch.rand(m, 1, device="cuda") / 64
        sw = torch.rand(n, 1, device="cuda") / 64
        out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        line = f"M{m:>5} N{n} K{k}: "

        # 1) aiter CK int8 (gfx90a JIT build)
        try:
            import aiter  # noqa: PLC0415

            dt = bench(lambda: aiter.gemm_a8w8(x8, w8, sx, sw.view(1, n), out))
            line += f"aiter-CK {tops(m,n,k,dt):7.2f} TOPS ({tops(m,n,k,dt)/PEAK_TOPS*100:4.1f}% peak)  "
        except Exception as exc:  # noqa: BLE001
            line += f"aiter-CK n/a ({type(exc).__name__})  "

        # 2) vendor library int8 (hipBLASLt via torch._int_mm)
        try:
            dt = bench(lambda: torch._int_mm(x8, w8.t().contiguous()))
            line += f"lib-int8 {tops(m,n,k,dt):7.2f} TOPS ({tops(m,n,k,dt)/PEAK_TOPS*100:4.1f}%)"
        except Exception as exc:  # noqa: BLE001
            line += f"lib-int8 n/a ({type(exc).__name__})"
        print(line, flush=True)

    # 3) Triton int8 matmul (does the AMD backend use MFMA int8?)
    try:
        import triton  # noqa: F401, PLC0415
        import triton.language as tl  # noqa: PLC0415

        @triton.jit
        def int8_matmul(a_ptr, b_ptr, c_ptr, M, N, K,
                        sam, sak, sbk, sbn, scm, scn,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            rm = pid_m * BM + tl.arange(0, BM)
            rn = pid_n * BN + tl.arange(0, BN)
            rk = tl.arange(0, BK)
            acc = tl.zeros((BM, BN), dtype=tl.int32)
            for k0 in range(0, K, BK):
                a = tl.load(a_ptr + rm[:, None] * sam + (k0 + rk)[None, :] * sak)
                b = tl.load(b_ptr + (k0 + rk)[:, None] * sbk + rn[None, :] * sbn)
                acc += tl.dot(a, b, out_dtype=tl.int32)
            tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc)

        M = N = K = 2048
        a = torch.randint(-127, 128, (M, K), dtype=torch.int8, device="cuda")
        b = torch.randint(-127, 128, (K, N), dtype=torch.int8, device="cuda")
        c = torch.empty((M, N), dtype=torch.int32, device="cuda")
        grid = (M // 64, N // 64)
        fn = lambda: int8_matmul[grid](a, b, c, M, N, K, a.stride(0), a.stride(1),
                                       b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                                       BM=64, BN=64, BK=64)
        dt = bench(fn)
        print(f"triton int8 {M}x{N}x{K}: {tops(M,N,K,dt):7.2f} TOPS "
              f"({tops(M,N,K,dt)/PEAK_TOPS*100:4.1f}% peak)")
    except Exception as exc:  # noqa: BLE001
        print(f"triton int8 n/a: {type(exc).__name__}: {str(exc)[:120]}")


if __name__ == "__main__":
    main()
