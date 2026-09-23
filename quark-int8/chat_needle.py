#!/usr/bin/env python3
"""chat 模式下的长上下文大海捞针（本路线相对 llama.cpp 16K 上限的独有价值所在）。
用法: python3 chat_needle.py [port] [target_tokens]
"""
import json, sys, time, urllib.request
PORT = sys.argv[1] if len(sys.argv) > 1 else "8119"
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
BASE = f"http://127.0.0.1:{PORT}"

def chat(user, n=64, thinking=False):
    payload = {"model":"/models","messages":[{"role":"user","content":user}],
               "max_tokens":n,"temperature":0,"chat_template_kwargs":{"thinking":thinking}}
    r = urllib.request.Request(BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(r, timeout=1800).read())
    return d, time.time() - t0

filler = ("The quarterly logistics report covers warehousing, freight and customs. " * 60)
needle = "\nIMPORTANT: the vault access code is ZEPHYR-7781.\n"
body = filler
while len(body) < TARGET * 4:
    body += filler
body = body[: TARGET * 4] + needle + filler[: 400]
q = ("Read the text below and answer with ONLY the vault access code.\n\n"
     + body + "\n\nQuestion: What is the vault access code?")
try:
    d, dt = chat(q, n=48)
    m = d["choices"][0]["message"]
    u = d.get("usage", {})
    print(f"prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')} 用时 {dt:.1f}s")
    print(f"content   = {m.get('content')!r}")
    if m.get("reasoning"): print(f"reasoning = {m['reasoning'][:150]!r}")
    ok = "ZEPHYR" in (m.get("content") or "").upper() or "7781" in (m.get("content") or "")
    print(f"捞到针: {'✅ 是' if ok else '❌ 否'}")
except Exception as e:
    print("失败:", type(e).__name__, str(e)[:200])
