#!/usr/bin/env python3
"""ctx 阶梯探针：固定确定性 prompt 在多个上下文档位上量 step_ms / 解码 TPS / accept。

方法论沿用 step_probe.py 的 /metrics 计数器差分（2026-09-24），并针对长 prompt 做两点修正：
  ① 计数器差分跨**整个请求**（服务器空闲前提下，Δdraft/SPEC = 精确步数，与接受率无关）；
  ② 步时用**流式首末 chunk 之间的解码窗**（last−first），把 tokenize/prefill 排除在
     step_ms 之外——长 prompt 的 tokenize 随 ctx 增长，不排除会污染"步时 ∝ ctx"的斜率。
  ③ TTFT = 首 chunk − 发送时刻（含 tokenize+prefill+第一步；warmup 后 prefill 走前缀缓存）。

每个档位：两遍校准把 prompt 调到目标 token 数（±2%），warmup 一发（付冷 prefill，丢弃），
再跑 N 发取中位数。prompt 内容确定性生成（无 RNG）⇒ 跨 boot 同 ctx 同内容，接受率可比。

用法: ctx_ladder.py PORT TAG SPEC [LEVELS_K=8,32,64] [N=3] [MAXTOK=64]
输出: 每档位一行 JSON（便于回填 launcher 头注释/配方）。
⚠️ 前提：跑探针期间服务器上没有其它并发请求（计数器差分是全局的）。
"""
import json
import re
import statistics
import sys
import time
import urllib.request

PORT = int(sys.argv[1])
TAG = sys.argv[2]
SPEC = int(sys.argv[3])
LEVELS_K = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "8,32,64").split(",")]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 3
MAXTOK = int(sys.argv[6]) if len(sys.argv) > 6 else 64
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


def counters():
    txt = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=60).read().decode()
    out = {}
    for key in ("spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total"):
        m = re.search(rf"^(?:vllm:)?{key}\{{[^}}]*\}}\s+([0-9.eE+]+)$", txt, re.M)
        out[key] = float(m.group(1)) if m else 0.0
    return out


print(f"# ctx_ladder port={PORT} tag={TAG} SPEC={SPEC} levels={LEVELS_K}K n={N} maxtok={MAXTOK}",
      flush=True)
for lvl in LEVELS_K:
    target = lvl * 1000
    prompt, ptok = calibrate(target)
    _u, ttft_cold, _s = _send_stream(prompt, MAXTOK)     # warmup：付冷 prefill，丢弃
    rows = []
    for _ in range(N):
        c0 = counters() if SPEC > 0 else None
        u, ttft, span = _send_stream(prompt, MAXTOK)
        ntok = u["completion_tokens"]
        if SPEC > 0:
            c1 = counters()
            drafted = c1["spec_decode_num_draft_tokens_total"] - c0["spec_decode_num_draft_tokens_total"]
            steps = drafted / SPEC
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
    acc = (med("tok_per_step") - 1) / SPEC if SPEC > 0 else 0.0
    rec = {
        "tag": TAG, "spec": SPEC, "level_k": lvl, "prompt_tokens": rows[-1]["prompt_tokens"],
        "step_ms_med": round(med("step_ms"), 2),
        "step_ms_spread_pct": round((max(r["step_ms"] for r in rows) -
                                     min(r["step_ms"] for r in rows)) / med("step_ms") * 100, 1),
        "tok_per_step": round(med("tok_per_step"), 3), "accept_pct": round(acc * 100, 1),
        "decode_tps_med": round(med("tps"), 2), "ttft_s_med": round(med("ttft_s"), 2),
        "ttft_cold_s": round(ttft_cold, 2), "n": N,
    }
    print(json.dumps(rec, ensure_ascii=False), flush=True)
