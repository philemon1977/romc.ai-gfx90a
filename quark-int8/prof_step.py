#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线单步 profile（在 rocprofv3 kernel-trace 下运行）。

为什么不用 torch.profiler：vLLM v1 把模型跑在**独立 worker 进程**里，driver 进程的
profiler 只能看到 8.8 us 的 hipDeviceSynchronize（2026-09-20 实测）。所以改由
rocprofv3 在系统层抓 kernel，本脚本只负责：
  ① 起模型、预热；② 用两个 **marker kernel** 夹住"要计时的那次 generate"；
  ③ 打印这段的 wall time。之后用 analyze_ktrace_window.py 按 marker 切窗口聚合。

用法（一次性容器，8 卡空闲）：
    python3 prof_step.py --ctx 8192 --gen 32
"""
import argparse
import os
import time

import torch
import triton
import triton.language as tl
from vllm import LLM, SamplingParams


@triton.jit
def _mi250_prof_marker(OUT, V: tl.constexpr):
    """窗口标记：trace 里找这个 kernel 名即可切出计时窗口（V=1 开始，V=2 结束）。"""
    tl.store(OUT + tl.program_id(0), V)


_MB = None


def mark(v):
    global _MB
    if _MB is None:
        _MB = torch.zeros(8, device="cuda", dtype=torch.int32)
    _mi250_prof_marker[(8,)](_MB, V=v)
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--rows", type=int, default=28)
    ap.add_argument("--eager", type=int, default=0)
    ap.add_argument("--len", type=int, default=32768)
    ap.add_argument("--torchprof", type=int, default=0)
    a = ap.parse_args()
    kw = dict(model="/models", tensor_parallel_size=8, dtype="bfloat16",
              max_model_len=a.len, max_num_seqs=32, max_num_batched_tokens=2048,
              gpu_memory_utilization=0.97, decode_context_parallel_size=int(os.environ.get("MI250_DCP", "8")),
              disable_log_stats=True, enforce_eager=bool(a.eager))
    if not a.eager:
        kw["max_cudagraph_capture_size"] = 8
    if os.environ.get("MI250_FST_MAX_BATCH_MB"):
        kw["load_format"] = "fastsafetensors"
    t0 = time.perf_counter()
    llm = LLM(**kw)
    print("[prof] 引擎就绪 %.1f s" % (time.perf_counter() - t0), flush=True)
    tok = llm.get_tokenizer()
    ids = tok("The quick brown fox jumps over the lazy dog. " * 4000)["input_ids"][:a.ctx]
    print("[prof] ctx=%d tokens" % len(ids), flush=True)
    prompt = {"prompt_token_ids": ids}
    llm.generate([prompt], SamplingParams(max_tokens=4, temperature=0.0))   # 预热
    sp = SamplingParams(max_tokens=a.gen, temperature=0.0)
    mark(1)
    t0 = time.perf_counter()
    llm.generate([prompt], sp)
    dt = time.perf_counter() - t0
    mark(2)
    print("[prof] 窗口内：%.3f s / %d tok = %.1f ms/token = %.2f tok/s（含 prefill）"
          % (dt, a.gen, dt * 1000 / a.gen, a.gen / dt), flush=True)
    if a.torchprof:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            llm.generate([prompt], sp)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=a.rows), flush=True)


if __name__ == "__main__":
    main()
