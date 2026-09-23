#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agent 长上下文测具 v2（修 v1 的三处测具缺陷，见 DCP 评审 2026-09-20）。

v1 的缺陷（本轮实测）：
  1. doc() 假设 3.6 字符/token，实测 2.03 ⇒ 目标 16K 实到 9.2K（门限虚高 1.78 倍，我自己的锅）。
  2. 填充是同一句重复 ~1000 次 ⇒ 与 DSA top-2048 选择共振，针尖可能整段进不了注意力（测具失效）。
  3. 单点 HIT/MISS 无法区分"选择没选中"与"注意力/合并算错"。

v2 对策：目标 token 按实测 2.03 字符/token 生成并**回读 usage 自校准**；填充句逐条唯一；
针尖用易复述的数字码并做 2K/4K/8K/16K/32K 命中率曲线（一次起服跑完，省装载循环）；
可选 --save-prompt 落盘，供 analyze_idx_dump.py 判定"针尖是否落在 top-2048 选择集里"。

用法:
  python3 agent_bench2.py --port 8121 --model glm-5.3 --sweep 2048,4096,8192,16384 --save-prompt /tmp/p.txt
"""
import argparse, json, time, urllib.request

BASE = None
CHARS_PER_TOKEN = 2.03   # 实测标定值（3.6 是错的，见上）
CODE = "427109"          # 数字针尖：比 ZEPHYR-7781 更容易被逐字复述


def post(path, body, timeout=7200):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def stream(prompt, max_tokens, temperature=0.0, timeout=7200):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": temperature, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; n = 0; tlast = None; usage = None; pieces = []
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
                pieces.append(ch[0]["text"]); n += 1
                if ttft is None:
                    ttft = time.time() - t0
                tlast = time.time()
    dt = (tlast - t0 - ttft) if (tlast and ttft is not None) else 0.0
    return {"ttft": ttft or float("nan"), "n": n, "text": "".join(pieces),
            "decode_tps": ((n - 1) / dt) if dt > 0 and n > 1 else float("nan"),
            "total_s": (tlast - t0) if tlast else float("nan"), "usage": usage or {}}


REGIONS = ["north", "south", "east", "west", "central", "coastal", "highland", "delta"]
ITEMS = ["staffing", "procurement", "inventory turnover", "regional distribution",
         "freight cost", "warehouse yield", "supplier delay", "return rate"]


def doc(n_tokens, tag="DOC"):
    """逐条唯一的合成文档：句子按序号变化，避免与 top-k 选择共振。"""
    parts, i, chars, budget = [], 0, 0, int(n_tokens * CHARS_PER_TOKEN)
    while chars < budget:
        s = ("Record %s-%06d: the %s office reported %s of %d units in quarter Q%d. "
             % (tag, i, REGIONS[i % len(REGIONS)], ITEMS[(i * 3) % len(ITEMS)],
                (i * 7919) % 10000, 1 + (i % 4)))
        parts.append(s); chars += len(s); i += 1
    return "".join(parts)


def needle_answer(doc_text, max_tokens=24):
    """在文档中点插入数字针尖并提问；返回 (hit, text, usage, ttft, tps, prompt)。"""
    pos = len(doc_text) // 2
    line = "\nThe vault access code is %s.\n" % CODE
    long_doc = doc_text[:pos] + line + doc_text[pos:]
    p = ("Read the document and answer with the access code only.\n\n" + long_doc +
         "\n\nQuestion: what is the vault access code?\nAnswer:")
    r = stream(p, max_tokens)
    return (CODE in r["text"], r["text"].strip(), r["usage"], r["ttft"],
            r["decode_tps"], p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8121)
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--sweep", default="2048,4096,8192,16384",
                    help="针尖长度曲线（真实目标 token，逗号分隔）")
    ap.add_argument("--answer-tokens", type=int, default=24)
    ap.add_argument("--save-prompt", default="", help="落盘最后一次 prompt，供 idx dump 分析")
    ap.add_argument("--gate", type=int, default=16384, help="判为门限的那一档")
    a = ap.parse_args()
    global BASE, MODEL
    BASE = "http://127.0.0.1:%d" % a.port; MODEL = a.model

    sizes = [int(x) for x in a.sweep.split(",") if x.strip()]
    print("=== 针尖命中率曲线（真实 token 目标，填充逐条唯一，数字码 %s）===" % CODE)
    results = []
    for tgt in sizes:
        text = doc(tgt, tag="R")
        hit, ans, usage, ttft, tps, prompt = needle_answer(text, a.answer_tokens)
        actual = usage.get("prompt_tokens")
        # 自校准：实测偏差 >10% 就按实际比值重生成一次并复测
        if actual and abs(actual - tgt) / tgt > 0.10:
            global CHARS_PER_TOKEN
            CHARS_PER_TOKEN = CHARS_PER_TOKEN * (tgt / actual)
            text = doc(tgt, tag="R")
            hit, ans, usage, ttft, tps, prompt = needle_answer(text, a.answer_tokens)
            actual2 = usage.get("prompt_tokens")
            print("  [calib] 目标 %d：首测 %s → 校准(%.3f 字符/token)后 %s"
                  % (tgt, actual, CHARS_PER_TOKEN, actual2))
            actual = actual2
        results.append((tgt, actual, hit, ans, ttft, tps))
        print("  目标%-7d 实测%-7s TTFT=%7.2fs 解码=%5.2f tok/s %s 答案=%r"
              % (tgt, actual, ttft, tps, "HIT " if hit else "MISS", ans[:60]))
    if a.save_prompt:
        open(a.save_prompt, "w").write(prompt)
        print("  [saved] 最后一条 prompt → %s（定位针尖 token 位置用）" % a.save_prompt)
    print("=== 曲线汇总（效率：命中/总数；门限档 %d）===" % a.gate)
    hits = sum(1 for _, _, h, _, _, _ in results if h)
    for tgt, actual, hit, _, ttft, tps in results:
        print("  %-9d→%-8s %s TTFT=%7.2fs %5.2f tok/s"
              % (tgt, actual, "HIT " if hit else "MISS", ttft, tps))
    print("  合计 %d/%d 命中；门限档可见" % (hits, len(results)))


if __name__ == "__main__":
    main()
