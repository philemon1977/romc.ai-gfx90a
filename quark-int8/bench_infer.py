#!/usr/bin/env python3
"""Smoke + single-stream throughput against the served DSV4.1-INT4 endpoint."""
import json
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8119"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
PROMPT = (sys.argv[3] if len(sys.argv) > 3 else
          "Explain in detail how a transformer KV cache works.")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"


def post(payload, timeout=1200):
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


print("--- smoke: 17*19 must be 323 ---")
try:
    d = post({"model": "/models",
              "messages": [{"role": "user",
                            "content": "What is 17*19? Return only the integer."}],
              "temperature": 0, "max_tokens": 64,
              "chat_template_kwargs": {"thinking": False}})
    msg = d["choices"][0]["message"]
    print("  content:", repr((msg.get("content") or "")[:160]))
    print("  usage:", d.get("usage", {}).get("completion_tokens"), "completion tokens")
except Exception as e:
    print("  smoke FAILED:", type(e).__name__, str(e)[:300])

print(f"--- single-stream, greedy, ignore_eos, max_tokens={N} ---")
try:
    t0 = time.time()
    d = post({"model": "/models",
              "messages": [{"role": "user", "content": PROMPT}],
              "temperature": 0, "ignore_eos": True, "max_tokens": N,
              "chat_template_kwargs": {"thinking": False}})
    dt = time.time() - t0
    u = d.get("usage", {})
    ct = u.get("completion_tokens") or 0
    pt = u.get("prompt_tokens") or 0
    print(f"  prompt_tokens={pt} completion_tokens={ct} wall={dt:.2f}s")
    if ct:
        print(f"  >>> {ct / dt:.2f} tok/s (single stream)")
except Exception as e:
    print("  throughput FAILED:", type(e).__name__, str(e)[:300])
