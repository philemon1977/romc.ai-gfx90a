#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agent 会话场景压测：多轮前缀复用 + 长上下文 TTFT + 解码吞吐。

用法: python3 agent_bench.py --port 8121 --model glm-5.3 --doc-tokens 32768 --turns 4
      python3 agent_bench.py --port 8121 --model glm-5.3 --needle-tokens 131072
"""
import argparse, json, time, urllib.request

BASE = None

def post(path, body, timeout=7200):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def stream(prompt, max_tokens, temperature=0.0):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": temperature, "stream": True,
                       # 没这个字段流式响应不带 usage ⇒ 前缀缓存命中数取不到
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; n = 0; tlast = None; usage = None; pieces = []
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and ch[0].get("text"):
                pieces.append(ch[0]["text"])
                n += 1
                if ttft is None:
                    ttft = time.time() - t0
                tlast = time.time()
    dt = (tlast - t0 - ttft) if (tlast and ttft is not None) else 0.0
    return {"ttft": ttft or float("nan"), "n": n, "text": "".join(pieces),
            "decode_tps": ((n - 1) / dt) if dt > 0 and n > 1 else float("nan"),
            "total_s": (tlast - t0) if tlast else float("nan"), "usage": usage or {}}

def doc(n_tokens, tag="DOC"):
    """合成文档：按目标 token 数生成（英文 ≈3.6 字符/token）。"""
    unit = ("Section %s: the quarterly logistics report covers staffing, procurement, "
            "inventory turnover and regional distribution metrics in detail. ")
    chunk = unit % tag
    return chunk * max(1, int(n_tokens * 3.6 / len(chunk)))

def metrics():
    try:
        txt = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    except Exception:
        return {}
    keep = ("prefix_cache", "num_requests_running", "num_requests_waiting", "kv_cache_usage")
    out = {}
    for line in txt.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        for k in keep:
            if k in line.split("{")[0]:
                out[line.split()[0]] = line.split()[-1]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8121)
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--doc-tokens", type=int, default=32768)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--answer-tokens", type=int, default=64)
    ap.add_argument("--needle-tokens", type=int, default=0, help=">0 时额外做长上下文针尖检索")
    a = ap.parse_args()
    global BASE, MODEL
    BASE = "http://127.0.0.1:%d" % a.port; MODEL = a.model

    print("=== 多轮 agent 会话（首轮文档 ≈ %d token，共 %d 轮） ===" % (a.doc_tokens, a.turns))
    ctx = "You are a careful assistant. Read the document and answer precisely.\n\n" + doc(a.doc_tokens)
    results = []
    for t in range(1, a.turns + 1):
        q = "\n\nQuestion %d: what does the document say about staffing? Answer briefly.\nAnswer:" % t
        r = stream(ctx + q, a.answer_tokens)
        u = r.get("usage") or {}
        pt = u.get("prompt_tokens"); cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        results.append({"turn": t, "prompt_tokens": pt, "cached_tokens": cached, **{k: r[k] for k in ("ttft", "decode_tps", "n")}})
        print("  轮%-2d prompt=%-7s cached=%-7s TTFT=%8.3fs  解码=%5.2f tok/s  出%3d token"
              % (t, pt, cached, r["ttft"], r["decode_tps"], r["n"]))
        # 把上一轮答案并入上下文，模拟 agent 多轮
        ctx = ctx + q + "(see previous answer)"
    if len(results) >= 2 and results[0]["ttft"]:
        print("  ⇒ 第 2 轮 TTFT / 第 1 轮 TTFT = %.3f （前缀缓存效果）" % (results[1]["ttft"] / results[0]["ttft"]))

    if a.needle_tokens:
        code = "ZEPHYR-7781"
        filler = doc(a.needle_tokens, tag="NEEDLE")
        pos = len(filler) // 2
        long_doc = filler[:pos] + ("\nThe access code is %s.\n" % code) + filler[pos:]
        p = ("Read the document and answer with the access code only.\n\n" + long_doc +
             "\n\nQuestion: what is the access code?\nAnswer:")
        print("=== 长上下文针尖（目标 ≈ %d token） ===" % a.needle_tokens)
        r = stream(p, 16)
        hit = code in r["text"]
        print("  prompt=%s TTFT=%.2fs 解码=%.2f tok/s 答案=%r [%s]"
              % (r["usage"].get("prompt_tokens"), r["ttft"], r["decode_tps"], r["text"].strip(), "HIT" if hit else "MISS"))
    print("=== 服务器指标 ===")
    for k, v in sorted(metrics().items()):
        print("  %-52s %s" % (k, v))

main()
