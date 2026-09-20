#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GEMV 微基准：**真实 checkpoint 布局**（gs=32）下量有效带宽与旋钮。

布局（实测自 model-00099-of-00141.safetensors，layer 6）：
  gate_proj.weight_packed  I32 [2048, 768] → uint8 [2048, 3072]   (K=6144, N=2048)
  gate_proj.weight_scale   BF16 [2048, 192]                       (group=32)
  up_proj  同形；down_proj I32 [6144, 256] → uint8 [6144, 1024], scale BF16 [6144, 64]

用法（容器内，空闲卡）：python3 moe_gemv_bench.py --experts 8 --tokens 1,8,32
"""
import argparse, json, os, struct, sys, time
import torch

MODEL = "/models"


def load_expert_tensors(layer=6, experts=8):
    """从 checkpoint 直接取真实权重（保证生产布局：int32 packed + bf16 scale + gs=32）。"""
    import glob
    from safetensors import safe_open
    files = sorted(glob.glob(os.path.join(MODEL, "*.safetensors")))
    want = {}
    for e in range(experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suf in ("weight_packed", "weight_scale"):
                want[f"model.layers.{layer}.mlp.experts.{e}.{proj}.{suf}"] = None
    for f in files:
        with safe_open(f, framework="pt") as fh:
            keys = [k for k in fh.keys() if k in want and want[k] is None]
            for k in keys:
                want[k] = fh.get_tensor(k)
        if all(v is not None for v in want.values()):
            break
    missing = [k for k, v in want.items() if v is None]
    if missing:
        sys.exit("缺张量: %s ..." % missing[:3])
    return want


def build(layer=6, experts=8, dev="cuda"):
    w = load_expert_tensors(layer, experts)
    def cat(name, proj_list):
        return torch.stack([torch.cat([w[f"model.layers.{layer}.mlp.experts.{e}.{p}.{name}"] for p in proj_list], dim=0) for e in range(experts)]).to(dev)
    B1 = cat("weight_packed", ["gate_proj", "up_proj"])   # [E, 4096, 768] int32
    S1 = cat("weight_scale", ["gate_proj", "up_proj"])    # [E, 4096, 192] bf16
    B2 = cat("weight_packed", ["down_proj"])              # [E, 6144, 256] int32
    S2 = cat("weight_scale", ["down_proj"])               # [E, 6144, 64] bf16
    B1u = B1.view(torch.uint8).contiguous()               # [E, 4096, 3072] uint8
    B2u = B2.view(torch.uint8).contiguous()               # [E, 6144, 1024] uint8
    B1u = B1u.transpose(1, 2).contiguous()                # [E, K/2=3072, N=4096]
    B2u = B2u.transpose(1, 2).contiguous()
    S1t = S1.transpose(1, 2).contiguous()
    S2t = S2.transpose(1, 2).contiguous()
    return B1u, S1t, B2u, S2t


def bench(fn, iters=20, warmup=5):
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
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--tokens", default="1,8,32")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--layer", type=int, default=6)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    if not os.path.isdir(MODEL):
        sys.exit("模型目录不在容器内：%s（起容器时请挂载 /models）" % MODEL)
    B1, S1, B2, S2 = build(a.layer, a.experts, dev)
    K1, N1 = B1.shape[1] * 2, B1.shape[2]
    K2, N2 = B2.shape[1] * 2, B2.shape[2]
    print("gemm1: K=%d N=%d  gemm2: K=%d N=%d  experts=%d gs=%d" % (K1, N1, K2, N2, a.experts, K1 // S1.shape[1]))
    print("每专家每 rank 字节: gemm1 w=%.2fMB s=%.2fMB | gemm2 w=%.2fMB s=%.2fMB" % (
        B1[0].numel() / 2**20, S1[0].numel() * 2 / 2**20, B2[0].numel() / 2**20, S2[0].numel() * 2 / 2**20))
    sys.path.insert(0, "/patches/moe_gemv")
    try:
        import mi250_moe_gemv_gs as G
    except Exception as e:
        sys.exit("导入 GEMV 补丁失败: %r" % (e,))
    print("已导入 GEMV 补丁:", G.__file__)
    print()
    print("== 现有内核：扫描 (BLOCK_N, BLOCK_K, num_warps) ==")
    for M in [int(x) for x in a.tokens.split(",")]:
        pairs = M * a.topk
        A1 = torch.randn(pairs, K1, device=dev, dtype=torch.bfloat16)
        ids = torch.randint(0, a.experts, (pairs,), device=dev, dtype=torch.int32)
        wts = torch.rand(pairs, device=dev, dtype=torch.float32)
        O1 = torch.empty(pairs, N1, device=dev, dtype=torch.bfloat16)
        bytes1 = (B1.numel() + S1.numel() * 2) / a.experts * a.experts  # 简化：全部专家都读一遍
        for bn in (64, 128, 256):
            for bk in (32, 64, 128):
                for nw in (4, 8):
                    grid = (pairs, N1 // bn)
                    def run():
                        G._gemv_moe_k[grid](A1, B1, S1, O1, ids, wts, M, K1, N1, a.topk,
                                            APPLY_W=True, GROUP=K1 // S1.shape[1],
                                            BLOCK_N=bn, BLOCK_K=bk, num_warps=nw)
                    try:
                        dt = bench(run)
                    except Exception as e:
                        print("  M=%-3d BN=%-3d BK=%-3d w=%d  ERR %s" % (M, bn, bk, nw, type(e).__name__))
                        continue
                    gbs = bytes1 / dt / 1e9 if dt > 0 else float("nan")
                    print("  M=%-3d(pairs %-3d) BN=%-3d BK=%-3d w=%d  %8.1f us  %7.1f GB/s" % (M, pairs, bn, bk, nw, dt * 1e6, gbs))
    print()
    print("参考：MI250 单 GCD HBM ≈ 1600 GB/s（我们的目标是往这个数的 30–50% 靠）")


if __name__ == "__main__":
    main()
