#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""速度探针：单流 TTFT / 解码 tok/s（多上下文长度）+ 并发聚合吞吐。不测质量。"""
import argparse, json, threading, time, urllib.request

BASE = None; MODEL = None


def one(prompt, max_tokens, temperature=0.0, timeout=7200):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": temperature, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; n = 0; tlast = None; usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
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
                n += 1
                if ttft is None:
                    ttft = time.time() - t0
                tlast = time.time()
    dt = (tlast - t0 - ttft) if (tlast and ttft is not None) else 0.0
    return {"ttft": ttft, "n": n, "total": (tlast - t0) if tlast else None,
            "decode_tps": ((n - 1) / dt) if dt > 0 and n > 1 else None, "usage": usage or {}}


def filler(n_tok):
    unit = "The quick brown fox jumps over the lazy dog and keeps running through the field. "
    return unit * max(1, int(n_tok * 2.03 / len(unit)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8122)
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--lens", default="1024,4096,16384")
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--conc", type=str, default="4", help="并发档位，逗号分隔，如 1,4,8,16,32")
    a = ap.parse_args()
    global BASE, MODEL
    BASE = "http://127.0.0.1:%d" % a.port; MODEL = a.model
    print("== 单流：TTFT / 解码 ==", flush=True)
    for L in [int(x) for x in a.lens.split(",")]:
        p = filler(L) + "\n\nQuestion: summarize in one sentence.\nAnswer:"
        r = one(p, a.gen)
        pt = r["usage"].get("prompt_tokens")
        print("  prompt=%-7s TTFT=%7.2fs 解码=%5.2f tok/s 出%3d token (总 %.2fs)"
              % (pt, r["ttft"] or float("nan"), r["decode_tps"] or float("nan"), r["n"], r["total"] or 0), flush=True)
    concs = [int(x) for x in str(a.conc).split(",") if x.strip() and int(x) > 0]
    if concs:
        p = filler(4096) + "\n\nQuestion: summarize.\nAnswer:"
        for c in concs:
            res = []
            t0 = time.time()
            ths = [threading.Thread(target=lambda: res.append(one(p, a.gen))) for _ in range(c)]
            [t.start() for t in ths]; [t.join() for t in ths]
            wall = time.time() - t0
            ok = [r for r in res if r["n"]]
            tot = sum(r["n"] for r in ok)
            dtps = [r["decode_tps"] for r in ok if r["decode_tps"]]
            print("  M=%-3d (%2d 路) %d token 用时 %5.1fs ⇒ 聚合 %6.2f tok/s  单流均值 %5.2f  成功 %d/%d"
                  % (c, c, a.gen, wall, tot / wall if wall else 0,
                     (sum(dtps) / len(dtps)) if dtps else 0, len(ok), c), flush=True)


if __name__ == "__main__":
    main()
