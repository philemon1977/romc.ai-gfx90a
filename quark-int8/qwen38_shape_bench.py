#!/usr/bin/env python3
"""8107 真实 decode 形状的内核级基准（**图内 replay 口径**，不加载 360 GB 模型）。

为什么要图内口径：本会话实测同一 GEMM **eager 46 µs/次 vs 图内 11.8 µs/次（差 4 倍）**，
eager 的 host 开销会把小 GEMM 的成本高估数倍 ⇒ 只用 eager 会得出反向结论。

形状取自 checkpoint 的真实权重（TP8 每 rank；见脚本尾部注释的来源），
判据：**达成带宽 / 1.638 TB/s**（MI250X 每 GCD）—— 权重主导形状离地板越远，越说明"内核占用率"有空间。
用法：qwen38_shape_bench.py [--moe] [--iters N]
"""
import argparse
import sys
import time

import torch

HBM_GBPS = 1638.0     # 每 GCD 实测可用带宽口径（前序 docs 用 1.638 TB/s）

# (名字, N, K) —— TP8 每 rank
SHAPES = [
    ("lm_head",          31040, 2560),
    ("gdn_in_proj_qkv",   1280, 2560),
    ("gdn_in_proj_z",      768, 2560),
    ("gdn_out_proj",      2560,  768),
    ("attn_q_proj",        768, 2560),
    ("attn_kv_proj",       512, 2560),
    ("attn_o_proj",       2560,  768),
    ("hc_mix_up",         1280,  320),
    ("hc_mix_down",        320, 1280),
    ("shared_gate_up",     160, 2560),
    ("shared_down",       2560,   80),
]
MS = [1, 2, 4, 6, 16]


def in_graph_us(fn, reps=20, replay=60):
    """把 reps 次调用捕获进一张图，再 replay 多次取每调用 µs（去掉 host 开销）。"""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(replay):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / replay / reps * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moe", action="store_true", help="额外跑 vLLM 的 bf16 Triton fused MoE")
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    print(f"device={torch.cuda.get_device_name(0)}  口径=图内 replay  地板={HBM_GBPS:.0f} GB/s/GCD")
    print(f"{'shape':<18}{'M':>4}{'N':>7}{'K':>6}{'权重MB':>8}{'µs':>9}{'GB/s':>8}{'占地板':>8}")
    for name, n, k in SHAPES:
        w = torch.empty(n, k, dtype=torch.bfloat16, device="cuda")
        w.normal_(0, 0.02)
        wbytes = n * k * 2
        for m in MS:
            x = torch.empty(m, k, dtype=torch.bfloat16, device="cuda")
            x.normal_(0, 0.02)
            fn = lambda x=x, w=w: torch.nn.functional.linear(x, w)
            try:
                us = in_graph_us(fn, reps=a.iters)
            except Exception as e:
                print(f"{name:<18}{m:>4}{n:>7}{k:>6}  图捕获失败: {type(e).__name__}")
                continue
            bw = wbytes / (us * 1e-6) / 1e9
            print(f"{name:<18}{m:>4}{n:>7}{k:>6}{wbytes/1e6:>8.1f}{us:>9.1f}{bw:>8.0f}{100*bw/HBM_GBPS:>7.0f}%",
                  flush=True)
        del w
        torch.cuda.empty_cache()

    if a.moe:
        print("\n--- vLLM bf16 fused MoE（真实运行内核；TP8 ⇒ 每 rank 64 专家）---")
        try:
            from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        except Exception as e:
            print(f"  跳过：无法导入 fused_experts（{type(e).__name__}: {e}）")
            return 0
        E, inter, hid, topk = 64, 640, 2560, 10
        w1 = torch.empty(E, 2 * inter, hid, dtype=torch.bfloat16, device="cuda").normal_(0, 0.02)
        w2 = torch.empty(E, hid, inter, dtype=torch.bfloat16, device="cuda").normal_(0, 0.02)
        wb = (w1.numel() + w2.numel()) * 2
        for m in (1, 6, 16, 64):
            x = torch.empty(m, hid, dtype=torch.bfloat16, device="cuda").normal_(0, 0.02)
            tw = torch.rand(m, topk, dtype=torch.float32, device="cuda")
            ti = torch.randint(0, E, (m, topk), dtype=torch.int32, device="cuda")
            fn = lambda x=x, tw=tw, ti=ti: fused_experts(x, w1, w2, tw, ti)
            try:
                us = in_graph_us(fn, reps=max(4, a.iters // 4))
            except Exception as e:
                print(f"  E={E} m={m}: 图捕获失败 {type(e).__name__}: {str(e)[:80]}")
                continue
            print(f"  fused_moe m={m:<4} 权重{wb/1e6:6.0f} MB  {us:8.1f} µs  达成 {wb/(us*1e-6)/1e9:5.0f} GB/s "
                  f"({100*(wb/(us*1e-6)/1e9)/HBM_GBPS:.0f}% 地板)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
