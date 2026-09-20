#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 sweep + 对拍（gemm1/gemm2，M 阶梯）。"""
import argparse, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_gemv_bench import bench                 # noqa: E402
from moe_gemv_cmp import build, ref_out, rel_err  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--tokens", default="1")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--which", default="1,2")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bn", default="32,64,128")
    ap.add_argument("--g", default="1,2,4,8")
    ap.add_argument("--warps", default="4,8")
    a = ap.parse_args()
    torch.manual_seed(0)
    B1, S1, B2, S2 = build(6, a.experts)
    sys.path.insert(0, "/patches/moe_gemv")
    import mi250_moe_gemv_gs as V1
    import mi250_moe_gemv_v3 as V3
    for which, (B, S) in (("1", (B1, S1)), ("2", (B2, S2))):
        if which not in a.which.split(","):
            continue
        E, N, Kh = B.shape
        K = Kh * 2
        gs = K // S.shape[2]
        nbytes = B.numel() + S.numel() * 2
        print()
        print("== gemm%s: K=%d N=%d gs=%d 全专家 %.1f MB ==" % (which, K, N, gs, nbytes / 2**20))
        for M in [int(x) for x in a.tokens.split(",")]:
            pairs = M * a.topk
            X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            ids = torch.arange(pairs, device="cuda", dtype=torch.int32) % a.experts
            wts = torch.rand(pairs, device="cuda", dtype=torch.float32) * 0.5 + 0.25
            O3 = torch.empty(pairs, N, device="cuda", dtype=torch.bfloat16)
            r = ref_out(B, S, X, ids, wts, gs, npairs=pairs) if a.check else None
            best = (1e9, (0, 0, 0))
            print("  M=%d (pairs=%d)" % (M, pairs))
            for bn in [int(x) for x in a.bn.split(",")]:
                if bn > N or N % bn:
                    continue
                for g in [int(x) for x in a.g.split(",")]:
                    if g * gs > K:
                        continue
                    for nw in [int(x) for x in a.warps.split(",")]:
                        grid = (pairs, N // bn)
                        try:
                            def run():
                                V3._gemv_moe_v3[grid](X, B, S, O3, ids, wts, M, K, N, a.topk,
                                                      APPLY_W=True, GROUP=gs, G_PER_STEP=g,
                                                      BLOCK_N=bn, num_warps=nw)
                            with torch.no_grad():
                                dt = bench(run, iters=20 if pairs <= 64 else 10, warmup=3)
                        except Exception as exc:
                            print(("    BN=%-4d G=%-2d w=%d  ERR %r" % (bn, g, nw, exc))[:160])
                            continue
                        err = rel_err(O3.float(), r) if a.check else float("nan")
                        tag = "    BN=%-4d G=%-2d BK=%-4d w=%d %9.1f us  %7.1f GB/s" % (
                            bn, g, g * gs, nw, dt * 1e6, nbytes / dt / 1e9)
                        if a.check:
                            tag += "  对拍 %.2f%%" % err
                        print(tag)
                        if dt < best[0]:
                            best = (dt, (bn, g, nw))
            print("    => v3 最好 %.1f us  BN=%d G=%d w=%d  (%.1f GB/s)" % (
                best[0] * 1e6, best[1][0], best[1][1], best[1][2], nbytes / best[0] / 1e9))
            for bn, bk, nw in [(64, 64, 4)]:
                grid = (pairs, N // bn)
                def run1():
                    V1._gemv_moe_k[grid](X, B, S, O3, ids, wts, M, K, N, a.topk, APPLY_W=True,
                                         GROUP=gs, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw)
                dt = bench(run1, iters=20 if pairs <= 64 else 10, warmup=3)
                print("    (v1 参考 BN=%d BK=%d w=%d: %.1f us)" % (bn, bk, nw, dt * 1e6))


if __name__ == "__main__":
    main()
