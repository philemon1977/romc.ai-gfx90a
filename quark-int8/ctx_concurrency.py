#!/usr/bin/env python3
"""并发摊薄探针：固定 ctx 下同时发 C 个请求，量「聚合 TPS / 单请求 TPS / 每步耗时」。

**为什么需要它**：8117 的验证批走的是回退核 `kernel_paged_attention_2d`，其 grid 是
`(num_seqs, num_kv_heads)` ⇒ 单流时每 die 只有 2 个 program 串行走 KV；**并发请求数直接
把并行度乘上去**。所以长 ctx 的这道税在单流下最重、并发下会被摊薄——本探针就是量这个摊薄曲线。

口径（与 ctx_ladder.py 同源）：
  per_req  = 每请求 completion_tokens / 解码窗 的中位数 ← **跨 C 比较以它为准（最干净）**
  agg_tps  = Σ completion_tokens / 墙钟（聚合吞吐；墙钟含 TTFT/尾部 ⇒ 偏保守）
  step_ms  = 墙钟 / (Δnum_drafts / C)
             ⚠️ **2026-09-24 实测更正**：`num_drafts` 是**每序列每步 +1**（上游 docstring 说
             "跨请求聚合"，与实测不符：C=4 时 Δ=46 而真实步数只有 ~11.4），所以必须除以
             **在跑的序列数**。但 KV 块不够时会有请求停在 `Waiting`（实测 C=6@64k 只有 ~4 个
             在跑）⇒ 这里的 step_ms 只能当**上界**读，别跨 C 硬比。
  scale_x  = agg_tps / per_req ≈ 实际达到的并发倍数（≈C ⇒ 线性摊薄，≈1 ⇒ 完全串行化）
  C 个请求用**同一固定 prompt**（前缀缓存共享，显存只需一份前缀）⇒ 内容受控、跨 boot 可比。

用法: ctx_concurrency.py PORT TAG SPEC CTX_K CONCS N MAXTOK
例:   ctx_concurrency.py 8117 conc-spec5 5 64 1,4,6 3 64
"""
import json
import statistics
import sys
import threading
import time

import ctx_ladder as cl


def burst(prompt, maxtok, c):
    """C 个线程同栅栏起跑；返回 (每请求 (tokens, ttft, span), 墙钟)。"""
    out = [None] * c
    barrier = threading.Barrier(c)

    def one(i):
        barrier.wait()
        u, ttft, span = cl._send_stream(prompt, maxtok)
        out[i] = (u["completion_tokens"], ttft, span)

    ths = [threading.Thread(target=one, args=(i,)) for i in range(c)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return out, time.time() - t0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cl.PORT = int(argv[0])
    tag, spec = argv[1], int(argv[2])
    ctx_k = int(argv[3])
    concs = [int(x) for x in argv[4].split(",")]
    n = int(argv[5]) if len(argv) > 5 else 3
    maxtok = int(argv[6]) if len(argv) > 6 else 64

    print(f"# ctx_concurrency port={cl.PORT} tag={tag} SPEC={spec} ctx={ctx_k}k concs={concs} "
          f"n={n} maxtok={maxtok}", flush=True)
    prompt, ptok = cl.calibrate(ctx_k * 1000)
    print(f"# prompt_tokens={ptok}", flush=True)
    cl._send_stream(prompt, 8)                       # warmup：把该 prompt 灌进前缀缓存（丢弃）

    for c in concs:
        rows = []
        for _ in range(n):
            c0 = cl.counters() if spec > 0 else None
            res, wall = burst(prompt, maxtok, c)
            if spec > 0:
                # num_drafts 是"每序列每步 +1"⇒ 折成调度步数要除以在跑序列数（此处按 C 近似）
                steps = cl.steps_between(c0, cl.counters(), spec) / max(c, 1)
            else:
                steps = max(r[0] for r in res)
            toks = sum(r[0] for r in res)
            per = [r[0] / r[2] for r in res if r[2] > 0]
            rows.append({"agg": toks / wall, "per": statistics.median(per),
                         "step_ms": wall / steps * 1000 if steps else float("nan")})
        med = lambda k: statistics.median(r[k] for r in rows)
        print(json.dumps({
            "tag": tag, "spec": spec, "ctx_k": ctx_k, "conc": c,
            "agg_tps_med": round(med("agg"), 2),
            "per_req_tps_med": round(med("per"), 2),
            "step_ms_med": round(med("step_ms"), 2),
            "scale_x": round(med("agg") / med("per"), 2) if med("per") else None,
            "n": n,
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
