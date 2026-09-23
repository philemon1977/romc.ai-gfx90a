#!/usr/bin/env python3
"""TTFT / 解码 TPS 分测（流式），对照 bench_infer.py 的非流式总时长。

用法: python3 bench_stream.py [PORT] [MAX_TOKENS] [PROMPT]

同时给出长上下文一次（默认 4 个长度档）：
  - TTFT（首 token 延迟，含 prefill）
  - decode TPS = (n-1) / (t_last - t_first)   ← 纯解码速率，不含 prefill
  - 总时长 / 端到端 TPS
"""
import json
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8119"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
PROMPT = (sys.argv[3] if len(sys.argv) > 3 else
          "Explain in detail how a transformer KV cache works.")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"


def stream(prompt, maxtok):
    payload = {"model": "/models",
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "ignore_eos": True, "max_tokens": maxtok,
               "stream": True, "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_first = None
    n = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                d = json.loads(body)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if t_first is None:
                        t_first = time.time()
                    n += 1
    t_end = time.time()
    return t0, t_first, t_end, n, usage


def run(label, prompt, maxtok):
    try:
        t0, tf, te, n, u = stream(prompt, maxtok)
    except Exception as e:
        print(f"  {label}: FAILED {type(e).__name__} {str(e)[:200]}")
        return
    pt = u.get("prompt_tokens")
    ct = u.get("completion_tokens") or n
    ttft = (tf - t0) if tf else float("nan")
    dec = (ct - 1) / (te - tf) if (tf and te > tf and ct > 1) else float("nan")
    print(f"  {label}: prompt_tokens={pt} completion={ct} "
          f"TTFT={ttft*1000:.0f} ms  decode={dec:.2f} tok/s  "
          f"端到端={ct/(te-t0):.2f} tok/s  (wall {te-t0:.2f}s)")


print(f"--- 流式分测 @{PORT} ---")
run("短提示", PROMPT, N)
# 长上下文 prefill：同一句话重复到 ~4k / ~16k tokens（粗略 1 token ≈ 4 字符英文）
run("长提示~4k", (PROMPT + " ") * 120, 64)
run("长提示~16k", (PROMPT + " ") * 480, 64)
