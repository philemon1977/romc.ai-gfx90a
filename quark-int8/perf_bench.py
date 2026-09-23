#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TTFT / TPS 压测 v2：不依赖流式解析（对 vLLM 的 SSE 差异更鲁棒）。

  TTFT  : 用 max_tokens=1 的非流式请求墙钟时间近似（含 prefill + 首 token）
  TPS   : 用 max_tokens=N 的墙钟时间减去同 ISL 的 TTFT，再算 (N-1)/(t-ttft)
  并发  : C 个并行非流式请求的聚合 tokens/墙钟
"""
import argparse, json, threading, time, urllib.request

BASE = None
MODEL = "/models"   # 由 --model 覆盖（GLM 服务名是 glm-5.3，写死 /models 会 404）

def req(prompt, max_tokens, stream=False):
    # ignore_eos：压测必须强制生成满 max_tokens，否则短答（如"What animal jumps?"→"dog"）
    # 会立刻 EOS，解码 TPS 变成 nan（2026-09-20 实测踩过）
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": stream, "ignore_eos": True,
            "min_tokens": max_tokens}
    if stream:
        # 不带这个字段，流式响应里没有 usage ⇒ prompt_tokens/cached_tokens 取不到
        body["stream_options"] = {"include_usage": True}
    d = json.dumps(body).encode()
    r = urllib.request.Request(BASE + "/v1/completions", d, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=3600) as resp:
        payload = json.loads(resp.read().decode())
    dt = time.time() - t0
    ch = payload["choices"][0]
    n = payload.get("usage", {}).get("completion_tokens", 0)
    pt = payload.get("usage", {}).get("prompt_tokens")
    return {"sec": dt, "n": n, "prompt_tokens": pt, "text": ch.get("text", "")[:60]}

def pad(isl):
    unit = "The quick brown fox jumps over the lazy dog near the river bank. "
    body = (unit * (isl // 8 + 8))[: max(64, int(isl * 4.2))]
    return "Read the text and answer with one word.\n\n" + body + "\n\nQuestion: What animal jumps?\nAnswer:"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8119)
    ap.add_argument("--model", default="/models", help="服务里的模型名（GLM 用 glm-5.3）")
    ap.add_argument("--isl", default="128,1024,4096")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--conc", default="1,4,8")
    ap.add_argument("--conc-isl", type=int, default=512)
    ap.add_argument("--conc-tokens", type=int, default=64)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    global BASE, MODEL
    MODEL = a.model
    BASE = "http://127.0.0.1:%d" % a.port
    res = {"label": a.label, "rows": [], "conc": []}
    print("== 单流 TTFT / TPS（TTFT 用 max_tokens=1 近似）==")
    for isl in [int(x) for x in a.isl.split(",")]:
        p = pad(isl)
        t1 = min(req(p, 1)["sec"] for _ in range(2))          # TTFT 近似
        r2 = req(p, a.max_tokens)
        dec = (r2["n"] - 1) / max(r2["sec"] - t1, 1e-6) if r2["n"] > 1 else float("nan")
        res["rows"].append({"isl": isl, "prompt_tokens": r2["prompt_tokens"], "ttft": t1,
                            "decode_tps": dec, "total_s": r2["sec"], "n": r2["n"]})
        print("  ISL≈%-5s prompt_tokens=%-6s TTFT=%7.3fs  解码=%5.2f tok/s  总=%6.2fs (%d tok)"
              % (isl, r2["prompt_tokens"], t1, dec, r2["sec"], r2["n"]))
    print("== 并发聚合（ISL≈%d, max_tokens=%d）==" % (a.conc_isl, a.conc_tokens))
    p = pad(a.conc_isl)
    for c in [int(x) for x in a.conc.split(",")]:
        out = [None] * c
        def work(i):
            out[i] = req(p, a.conc_tokens)
        ths = [threading.Thread(target=work, args=(i,)) for i in range(c)]
        t0 = time.time()
        for t in ths: t.start()
        for t in ths: t.join()
        wall = time.time() - t0
        tot = sum(z["n"] for z in out)
        rec = {"conc": c, "agg_tps": tot / wall, "wall_s": wall, "total_tokens": tot,
               "per_req_s": sum(z["sec"] for z in out) / c}
        res["conc"].append(rec)
        print("  conc=%-2d 聚合=%6.2f tok/s  墙钟=%6.2fs  合计 %d tok" % (c, rec["agg_tps"], wall, tot))
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
        print("RESULT_JSON=" + a.out)

main()
