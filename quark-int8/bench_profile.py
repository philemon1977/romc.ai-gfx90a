#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TPS 瓶颈定位：① 并发扩展性 ② torch profiler 抓窗口。
并发扩展性是判据：
  agg TPS 随并发近似线性 ⇒ 瓶颈是"每步固定开销"（launch/sync/串行 Python）
  agg TPS 不随并发涨   ⇒ 瓶颈是"单点串行"（全局锁 / 单线程循环）
"""
import argparse, json, time, urllib.request, threading

BASE = None
MODEL = None

def _req(path, body=None, timeout=7200):
    data = json.dumps(body).encode() if body is not None else b""
    r = urllib.request.Request(BASE + path, data, {"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=timeout).read().decode()

def gen(prompt, max_tokens):
    t0 = time.time()
    d = json.loads(_req("/v1/completions", {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                                            "temperature": 0.0, "stream": False}))
    dt = time.time() - t0
    u = d.get("usage", {}) or {}
    return dt, int(u.get("completion_tokens", 0)), int(u.get("prompt_tokens", 0))

def conc_test(n, prompt, max_tokens):
    res = [None] * n
    def work(i):
        try: res[i] = gen(prompt, max_tokens)
        except Exception as e: res[i] = ("ERR", type(e).__name__)
    ts = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.time() - t0
    ok = [r for r in res if r and r[0] != "ERR"]
    ntok = sum(r[1] for r in ok)
    return wall, ntok, len(ok), n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8121)
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    global BASE, MODEL
    BASE = "http://127.0.0.1:%d" % a.port; MODEL = a.model
    report = {}

    short = "Question: say hello.\nAnswer:"
    print("=== ① 短上下文单流 decode ===")
    dt, ct, pt = gen(short, 64)
    tps = (ct - 1) / dt if dt > 0 and ct > 1 else float("nan")
    print("  prompt=%d out=%d wall=%.2fs ⇒ 端到端 %.2f tok/s（含 prefill）" % (pt, ct, dt, ct / dt))
    report["single_stream_e2e_tps"] = ct / dt

    print("=== ② 并发扩展性（同一短提示，各 32 token）===")
    rows = []
    for n in (1, 2, 4):
        wall, ntok, ok, tot = conc_test(n, short, 32)
        agg = ntok / wall if wall > 0 else float("nan")
        rows.append((n, wall, ntok, agg))
        print("  并发=%d  wall=%6.2fs  出token=%4d  聚合=%6.2f tok/s  成功=%d/%d" % (n, wall, ntok, agg, ok, tot))
    report["concurrency"] = rows
    if rows and rows[0][3] > 0:
        for n, _, _, agg in rows:
            print("    并发%d 的扩展比 = %.2fx" % (n, agg / rows[0][3]))

    print("=== ③ torch profiler 抓取（1 个请求，24 token）===")
    try:
        _req("/start_profile")
        print("  start_profile OK")
        t0 = time.time()
        gen(short, 24)
        print("  采样 %.2fs" % (time.time() - t0))
        _req("/stop_profile")
        print("  stop_profile OK")
        report["profile"] = "ok"
    except Exception as e:
        print("  profiler 不可用：", type(e).__name__, e)
        report["profile"] = "fail: %s" % type(e).__name__

    if a.out:
        json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=2)
        print("report ->", a.out)

main()
