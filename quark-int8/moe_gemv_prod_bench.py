#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按**生产分片形状**调 v3：gemm1 K=6144 N=512、gemm2 K=2048 N=768（TP8 各切 1/8）。

为什么必须这样测：microbench 用全 N=4096 时网格是 8x64=512 program，跑出 275 GB/s；
生产里 N_local=512 ⇒ 网格只有 8x8=64 program（104 CU 的机器上严重欠占用），
真实效率要低得多。这里按真实形状 + 真实 pairs(8/16/32/64) 扫 BLOCK_N/G/num_warps。
"""
import argparse, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_gemv_bench import bench                 # noqa: E402
from moe_gemv_cmp import build, ref_out, rel_err  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--tokens", default="1,2,4,8")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bn", default="16,32,64,128")
    ap.add_argument("--g", default="2,4,8")
    ap.add_argument("--warps", default="1,2,4")
    a = ap.parse_args()
    torch.manual_seed(0)
    B1, S1, B2, S2 = build(6, a.experts)
    # TP8 分片：gemm1 N=4096/8=512，gemm2 N=6144/8=768
    shapes = [("gemm1", B1[:, :512].contiguous(), S1[:, :512].contiguous()),
              ("gemm2", B2[:, :768].contiguous(), S2[:, :768].contiguous())]
    sys.path.insert(0, "/patches/moe_gemv")
    import mi250_moe_gemv_gs as M
    import mi250_moe_gemv_v3 as V3
    for name, B, S in shapes:
        E, N, Kh = B.shape
        K = Kh * 2
        gs = K // S.shape[2]
        nbytes = B.numel() + S.numel() * 2
        print()
        print("== %s（TP8 分片）: K=%d N_local=%d gs=%d  每 token %.2f MB ==" % (
            name, K, N, gs, nbytes / 2**20))
        for M_ in [int(x) for x in a.tokens.split(",")]:
            pairs = M_ * a.topk
            X = torch.randn(M_, K, device="cuda", dtype=torch.bfloat16)
            ids = torch.arange(pairs, device="cuda", dtype=torch.int32) % a.experts
            wts = torch.rand(pairs, device="cuda", dtype=torch.float32) * 0.5 + 0.25
            O = torch.empty(pairs, N, device="cuda", dtype=torch.bfloat16)
            r = ref_out(B, S, X, ids, wts, gs, npairs=pairs) if a.check else None
            print("  M=%d (pairs=%d, grid=%d)" % (M_, pairs, pairs * max(1, N // 32)))
            rows = []
            for bn in [int(x) for x in a.bn.split(",")]:
                if N % bn:
                    continue
                for g in [int(x) for x in a.g.split(",")]:
                    if g * gs > K or K % (g * gs):
                        continue
                    for nw in [int(x) for x in a.warps.split(",")]:
                        grid = (pairs, N // bn)
                        try:
                            def run():
                                V3._gemv_moe_v3[grid](X, B, S, O, ids, wts, M_, K, N, a.topk,
                                                      APPLY_W=True, GROUP=gs, G_PER_STEP=g,
                                                      BLOCK_N=bn, num_warps=nw)
                            with torch.no_grad():
                                dt = bench(run, iters=20 if pairs <= 64 else 10, warmup=3)
                        except Exception as exc:
                            rows.append((1e9, bn, g, nw, "ERR %s" % type(exc).__name__))
                            continue
                        err = rel_err(O.float(), r) if a.check else None
                        rows.append((dt, bn, g, nw, ("%.2f%%" % err) if err is not None else ""))
            rows.sort()
            for dt, bn, g, nw, extra in rows[:8]:
                if dt >= 1e9:
                    print("    BN=%-4d G=%-2d w=%d  %s" % (bn, g, nw, extra))
                else:
                    print("    BN=%-4d G=%-2d w=%d %9.1f us  %7.1f GB/s %s" % (
                        bn, g, nw, dt * 1e6, nbytes / dt / 1e9, extra))
            if rows and rows[0][0] < 1e9:
                print("    => 最优 BN=%d G=%d w=%d" % (rows[0][1], rows[0][2], rows[0][3]))
            # v1 同形状对照
            for bn, bk, nw in [(64, 64, 4)]:
                grid = (pairs, N // bn)
                def run1():
                    M._gemv_moe_k[grid](X, B, S, O, ids, wts, M_, K, N, a.topk, APPLY_W=True,
                                        GROUP=gs, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw)
                dt = bench(run1, iters=20 if pairs <= 64 else 10, warmup=3)
                print("    (v1 BN=%d BK=%d w=%d: %.1f us  %.1f GB/s)" % (
                    bn, bk, nw, dt * 1e6, nbytes / dt / 1e9))


if __name__ == "__main__":
    main()
