#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TPS 探针（QR A/B 用）：给一个并发档，测聚合 tok/s 与单请求延迟。

口径（两臂完全一致）：
  · ISL≈800（固定填充文本）/ OSL=OSL 上限 / temp0 / 流式；
  · token 计数用 stream_options.include_usage 的真实 completion_tokens（不数 chunk）；
  · 聚合 = 总 completion_tokens / 墙钟（含所有请求的排队与解码）。

用法：python3 qr_tps_probe.py <port> <model> <conc> <num_req> [osl]
"""
import json
import sys
import threading
import time
import urllib.request

PORT = int(sys.argv[1]); MODEL = sys.argv[2]
CONC = int(sys.argv[3]); N = int(sys.argv[4])
OSL = int(sys.argv[5]) if len(sys.argv) > 5 else 128
# ISL 倍数（env 传入，不改臂脚本）。DCP / split-K 这类杠杆强烈依赖上下文长度：
# 短档判负不等于长档判负，必须两档都测。1 ≈ 700 token，8 ≈ 5.6k，24 ≈ 17k。
MULT = int(__import__("os").environ.get("TPS_ISL_MULT", "1") or 1)

FILLER = ("长江流经青海、西藏、四川、云南、重庆、湖北、湖南、江西、安徽、江苏、上海十一个省区市，"
          "全长约六千三百公里，是中国第一长河，也是世界第三长河。") * 6
PROMPT = "阅读下面这段文字，然后用一句话概括它说了什么。\n" + FILLER * MULT

lock = threading.Lock()
res = {"tok": 0, "lat": [], "err": 0}


def one():
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.0, "max_tokens": OSL, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    tok = 0
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for line in r:
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                j = json.loads(payload)
                if j.get("usage"):
                    tok = int(j["usage"].get("completion_tokens") or tok)
        with lock:
            res["tok"] += tok
            res["lat"].append(time.time() - t0)
    except Exception:
        with lock:
            res["err"] += 1


t_start = time.time()
threads = []
for i in range(N):
    while sum(1 for t in threads if t.is_alive()) >= CONC:
        time.sleep(0.02)
    t = threading.Thread(target=one)
    t.start()
    threads.append(t)
for t in threads:
    t.join()
wall = time.time() - t_start
lat = sorted(res["lat"])
p50 = lat[len(lat) // 2] if lat else 0.0
print(json.dumps({
    "conc": CONC, "num_req": N, "osl_cap": OSL,
    "agg_tps": round(res["tok"] / wall, 2) if wall else 0.0,
    "tok_total": res["tok"], "wall_s": round(wall, 2),
    "p50_lat_s": round(p50, 2), "err": res["err"],
}, ensure_ascii=False))
