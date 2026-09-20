#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GEMV 变体对拍台：现有内核 vs 二维累加器变体（真实 checkpoint 布局 [E,N,K/2]）。

背景：moe_gemv_bench.py 实测现有内核 M=1 时 gemm1 最好 2.02 ms / 113 MB = 56 GB/s，
而 75 层 x 2.02 ms ≈ 152 ms/token ≈ 6.6 tok/s，与端到端 6.8 tok/s 吻合 ⇒ gemm1 即主瓶颈。
现有内核每个 K 步都做一次 tl.sum(axis=1) 跨 lane 归约（48 步 x 2 j = 96 次/输出元素），
本台子量「二维累加器 + 收尾归约一次」能拿回多少。

用法：python3 moe_gemv_cmp.py --experts 8 --tokens 1,8,32 [--check]
"""
import argparse, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_gemv_bench import load_expert_tensors, bench   # noqa: E402

MODEL = "/models"


def build(layer=6, experts=8, dev="cuda"):
    """真实生产布局：[E, N, K//2] uint8（不做任何 transpose）。"""
    w = load_expert_tensors(layer, experts)

    def cat(name, projs):
        return torch.stack([torch.cat([w["model.layers.%d.mlp.experts.%d.%s.%s" % (layer, e, p, name)]
                                       for p in projs], dim=0) for e in range(experts)]).to(dev)
    B1 = cat("weight_packed", ["gate_proj", "up_proj"]).view(torch.uint8).contiguous()
    S1 = cat("weight_scale", ["gate_proj", "up_proj"]).contiguous()
    B2 = cat("weight_packed", ["down_proj"]).view(torch.uint8).contiguous()
    S2 = cat("weight_scale", ["down_proj"]).contiguous()
    return B1, S1, B2, S2


def ref_out(B, S, X, ids, wts, group, npairs=None):
    """torch 参考：逐专家反量化到 fp32 再乘。B[E,N,K/2] uint8, S[E,N,K/g] bf16, X[T,K] bf16。"""
    E, N, Kh = B.shape
    K = Kh * 2
    pairs = ids.numel() if npairs is None else npairs
    out = torch.zeros(pairs, N, device=X.device, dtype=torch.float32)
    Bi = B.to(torch.int32)
    for e in sorted(set(int(x) for x in ids[:pairs].tolist() if x >= 0)):
        lo = (Bi[e] & 0xF) - 8
        hi = ((Bi[e] >> 4) & 0xF) - 8
        Wd = torch.stack([lo, hi], dim=-1).reshape(N, K).float()          # [N, K] 逐 k 反量化
        Wd = Wd * S[e].float().repeat_interleave(group, dim=-1)            # 乘 group scale
        for p in range(pairs):
            if int(ids[p]) != e:
                continue
            t = p // max(1, pairs // X.shape[0]) if False else p // (pairs // X.shape[0])
            x = X[t].float()
            out[p] = (Wd @ x) * (wts[p] if wts is not None else 1.0)
    return out


def rel_err(a, b):
    return (a - b).abs().mean().item() / (b.abs().mean().item() + 1e-6) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--tokens", default="1,8,32")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--which", default="1,2", help="1=gemm1(gate+up) 2=gemm2(down)")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    if not os.path.isdir(MODEL):
        sys.exit("模型目录不在容器内：%s" % MODEL)
    B1, S1, B2, S2 = build(a.layer, a.experts, dev)
    sys.path.insert(0, "/patches/moe_gemv")
    import mi250_moe_gemv_gs as V1
    import mi250_moe_gemv_v2 as V2
    print("v1:", V1.__file__)
    print("v2:", V2.__file__)
    for which, (B, S) in (("1", (B1, S1)), ("2", (B2, S2))):
        if which not in a.which.split(","):
            continue
        E, N, Kh = B.shape
        K = Kh * 2
        group = K // S.shape[2]
        nbytes = B.numel() + S.numel() * 2
        print()
        print("== gemm%s: K=%d N=%d gs=%d 全专家字节 %.1f MB ==" % (which, K, N, group, nbytes / 2**20))
        for M in [int(x) for x in a.tokens.split(",")]:
            pairs = M * a.topk
            X = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            ids = torch.arange(pairs, device=dev, dtype=torch.int32) % a.experts
            wts = torch.rand(pairs, device=dev, dtype=torch.float32) * 0.5 + 0.25
            O1 = torch.empty(pairs, N, device=dev, dtype=torch.bfloat16)
            O2 = torch.empty(pairs, N, device=dev, dtype=torch.bfloat16)
            if a.check:
                r = ref_out(B, S, X, ids, wts, group, npairs=pairs)
                print("  参考: %s" % (tuple(r.shape),))
            rows = []
            # 最后一维必须能被 BLOCK_N 整除
            cands = [(64, 32, 4), (64, 64, 4), (128, 32, 4), (128, 64, 4), (128, 64, 8),
                     (128, 128, 8), (256, 64, 8), (256, 32, 4), (512, 32, 8), (1024, 32, 8)]
            cands = [(bn, bk, nw) for (bn, bk, nw) in cands if bn <= N and N % bn == 0]
            for bn, bk, nw in cands:
                grid = (pairs, N // bn)
                for tag, kern, O in (("v1", V1._gemv_moe_k, O1), ("v2", V2._gemv_moe_acc2_kernel, O2)):
                    try:
                        def run():
                            kern[grid](X, B, S, O, ids, wts, M, K, N, a.topk, APPLY_W=True,
                                       GROUP=group, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw)
                        it = 20 if pairs <= 64 else 10
                        with torch.no_grad():
                            dt = bench(run, iters=it, warmup=3)
                    except Exception as exc:
                        rows.append((tag, bn, bk, nw, float("nan"), type(exc).__name__))
                        continue
                    err = rel_err(O.float(), r) if a.check else float("nan")
                    rows.append((tag, bn, bk, nw, dt, err))
            print("  M=%d (pairs=%d)" % (M, pairs))
            for tag, bn, bk, nw, dt, err in rows:
                if dt != dt:
                    print("    %s BN=%-4d BK=%-3d w=%d   ERR %s" % (tag, bn, bk, nw, err))
                    continue
                print("    %s BN=%-4d BK=%-3d w=%d %9.1f us  %7.1f GB/s%s" % (
                    tag, bn, bk, nw, dt * 1e6, nbytes / dt / 1e9,
                    ("  对拍 %.2f%%" % err) if a.check else ""))
            # 提速比（拿两者各自最好配置）
            best = {}
            for tag, bn, bk, nw, dt, err in rows:
                if dt == dt and (tag not in best or dt < best[tag][0]):
                    best[tag] = (dt, bn, bk, nw)
            if "v1" in best and "v2" in best:
                print("    => 最好: v1 %.1f us (BN=%d BK=%d w=%d) | v2 %.1f us (BN=%d BK=%d w=%d) | 提速 %.2fx" % (
                    best["v1"][0] * 1e6, best["v1"][1], best["v1"][2], best["v1"][3],
                    best["v2"][0] * 1e6, best["v2"][1], best["v2"][2], best["v2"][3],
                    best["v1"][0] / best["v2"][0]))
    print()
    print("参考：MI250 单 GCD HBM ≈ 1600 GB/s；gemm1 每 token 每层读 %.1f MB" % ((B1.numel() + S1.numel() * 2) / 2**20))


if __name__ == "__main__":
    main()
