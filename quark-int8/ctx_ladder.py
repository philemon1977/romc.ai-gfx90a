#!/usr/bin/env python3
"""ctx 阶梯探针：固定确定性 prompt 在多个上下文档位上量 step_ms / 解码 TPS / accept。

方法论（2026-09-24 定稿）：
  ① 步数口径 = `vllm:spec_decode_num_drafts_total` 差分。
     ⚠️ **2026-09-24 实测更正**：该计数实际是**每序列每步 +1**（上游 `SpecDecodingStats` 的
     docstring 写"跨请求聚合"，与实测不符：并发 C=4 时 Δ=46，而日志用 accepted/drafted 吞吐
     反推的真实步数只有 ~11.4）⇒ **单流（C=1）下它 = 调度步数**，本脚本用法成立；
     并发场景请用 `ctx_concurrency.py`（按在跑序列数折算，且以内层 span 口径为准）。
  ② 步时用**流式首末 chunk 之间的解码窗**（last−first），把 tokenize/prefill 排除在 step_ms
     之外——长 prompt 的 tokenize 随 ctx 增长，不排除会污染"步时 ∝ ctx"的斜率。
  ③ TTFT = 首 chunk − 发送时刻（warmup 后 prefill 走前缀缓存）。

每个档位：两遍校准把 prompt 调到目标 token 数（±2%），warmup 一发（付冷 prefill，丢弃），
再跑 N 发取中位数。prompt 内容确定性生成（无 RNG）⇒ 跨 boot 同 ctx 同内容，接受率可比。

用法: ctx_ladder.py PORT TAG SPEC [LEVELS_K=8,32,64] [N=3] [MAXTOK=64]
输出: 每档位一行 JSON（便于回填 launcher 头注释/配方）。
⚠️ 前提：跑探针期间服务器上没有其它并发请求（差分是全局的）。
也可被 ctx_concurrency.py 导入复用（设置模块级 PORT 后调 calibrate/_send_stream/counters）。
"""
import json
import os
import re
import statistics
import sys
import time
import urllib.request

PORT = int(os.environ.get("CTX_LADDER_PORT", "8117"))
MODEL = "ornith"

# 确定性填充文本：12 句中性技术句轮转 + 行号前缀（避免纯重复文本的高接受假象）。
_SENTS = [
    "The attention backend selects kernels by block size, dtype and layout at startup.",
    "Quantized weights are dequantized on the fly inside the fused GEMM epilogue.",
    "Speculative decoding drafts several tokens per step and verifies them in one pass.",
    "The KV cache pool is sharded across tensor parallel ranks by attention heads.",
    "Chunked prefill splits long prompts into scheduler sized pieces before decode.",
    "Prefix caching reuses previously computed blocks instead of recomputing them.",
    "Linear attention keeps a fixed size recurrent state that does not grow with context.",
    "The allreduce cost per layer depends on world size and the chosen communication path.",
    "Kernel autotuning benchmarks candidate tile shapes the first time a shape appears.",
    "Memory fragmentation can prevent the allocator from returning freed segments.",
    "The scheduler aligns decode steps to block boundaries for hybrid state snapshots.",
    "Throughput numbers must carry their workload label or they cannot be compared.",
]
_CHARS_PER_TOK = 3.7   # 英文技术文本经验值；两遍校准自修正


def build_prompt(target_tokens: int, lines: int) -> str:
    return "\n".join(f"[ref {i:06d}] {_SENTS[i % len(_SENTS)]}" for i in range(lines))


def _send_stream(prompt: str, max_tokens: int):
    """流式一发：返回 (usage, ttft_s, decode_span_s)。"""
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "ignore_eos": True, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    t_send = time.time()
    resp = urllib.request.urlopen(req, timeout=1800)
    first = last = None
    usage = None
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        d = json.loads(payload)
        if d.get("usage"):
            usage = d["usage"]
        now = time.time()
        if first is None:
            first = now
        last = now
    resp.close()
    if usage is None or first is None:
        raise RuntimeError("流式响应缺 usage/内容 chunk")
    return usage, first - t_send, last - first


def calibrate(target_tokens: int):
    """两遍校准：先估行数，用真实 prompt_tokens 修正行数。"""
    lines = max(4, int(target_tokens * _CHARS_PER_TOK / 90))   # 每行 ~95 字符
    got = target_tokens
    for _ in range(3):
        u, _t, _s = _send_stream(build_prompt(target_tokens, lines), 8)
        got = u["prompt_tokens"]
        if abs(got - target_tokens) / target_tokens < 0.02:
            break
        lines = max(4, int(lines * target_tokens / got))
    return build_prompt(target_tokens, lines), got


_METRIC_KEYS = ("spec_decode_num_drafts_total",
                "spec_decode_num_draft_tokens_total",
                "spec_decode_num_accepted_tokens_total")


def counters():
    txt = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=60).read().decode()
    out = {}
    for key in _METRIC_KEYS:
        m = re.search(rf"^(?:vllm:)?{key}\{{[^}}]*\}}\s+([0-9.eE+]+)$", txt, re.M)
        out[key] = float(m.group(1)) if m else 0.0
    return out


def steps_between(c0, c1, spec):
    """两次取数之间的**序列-步数**（num_drafts 每序列每步 +1；C=1 时 = 调度步数）。"""
    steps = c1["spec_decode_num_drafts_total"] - c0["spec_decode_num_drafts_total"]
    if steps <= 0 and spec > 0:                     # 旧版/缺指标时回退
        steps = (c1["spec_decode_num_draft_tokens_total"]
                 - c0["spec_decode_num_draft_tokens_total"]) / spec
    return steps


def measure_level(spec: int, target_tokens: int, n: int, maxtok: int):
    """跑一个 ctx 档位，返回记录 dict。调用前需设好模块级 PORT。"""
    prompt, _ptok = calibrate(target_tokens)
    _u, ttft_cold, _s = _send_stream(prompt, maxtok)      # warmup：付冷 prefill，丢弃
    rows = []
    for _ in range(n):
        c0 = counters() if spec > 0 else None
        u, ttft, span = _send_stream(prompt, maxtok)
        ntok = u["completion_tokens"]
        if spec > 0:
            steps = steps_between(c0, counters(), spec)
        else:
            steps = ntok                                  # 无投机：1 tok/step
        span_steps = max(steps - 1, 1)                    # 首 chunk 落在第 1 步末
        rows.append({
            "prompt_tokens": u["prompt_tokens"], "steps": steps,
            "step_ms": span / span_steps * 1000,
            "tok_per_step": ntok / steps if steps else float("nan"),
            "tps": ntok / span if span else float("nan"),
            "ttft_s": ttft,
        })
    med = lambda k: statistics.median(r[k] for r in rows)
    acc = (med("tok_per_step") - 1) / spec if spec > 0 else 0.0
    return {
        "spec": spec, "level_k": target_tokens // 1000, "prompt_tokens": rows[-1]["prompt_tokens"],
        "step_ms_med": round(med("step_ms"), 2),
        "step_ms_spread_pct": round((max(r["step_ms"] for r in rows) -
                                     min(r["step_ms"] for r in rows)) / med("step_ms") * 100, 1),
        "tok_per_step": round(med("tok_per_step"), 3), "accept_pct": round(acc * 100, 1),
        "decode_tps_med": round(med("tps"), 2), "ttft_s_med": round(med("ttft_s"), 2),
        "ttft_cold_s": round(ttft_cold, 2), "n": n,
    }


def main(argv=None):
    global PORT
    argv = sys.argv[1:] if argv is None else argv
    PORT = int(argv[0])
    tag, spec = argv[1], int(argv[2])
    levels = [int(x) for x in (argv[3] if len(argv) > 3 else "8,32,64").split(",")]
    n = int(argv[4]) if len(argv) > 4 else 3
    maxtok = int(argv[5]) if len(argv) > 5 else 64
    print(f"# ctx_ladder port={PORT} tag={tag} SPEC={spec} levels={levels}K n={n} maxtok={maxtok}",
          flush=True)
    for lvl in levels:
        rec = measure_level(spec, lvl * 1000, n, maxtok)
        rec["tag"] = tag
        print(json.dumps(rec, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
