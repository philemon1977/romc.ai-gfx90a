#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B 质量评测：同一把尺子量两个 checkpoint，把"退化"变成可比的数字。

## 为什么需要它

"输出退化成复读"此前一直靠肉眼判断（"的的的的"）。要判定
"逐组最优 scale 是否救回质量"，必须有**可复现、可对比**的量尺，且两个 checkpoint
必须走**完全相同的**提示、参数与评分口径（否则就是两把尺子）。

## 用法（服务已起在指定端口）

    python3 eval_quality_ab.py --port 8119 --label old   --out /tmp/q_old.json
    python3 eval_quality_ab.py --port 8119 --label new   --out /tmp/q_new.json
    python3 eval_quality_ab.py --compare /tmp/q_old.json /tmp/q_new.json

## 指标（每个回答）

  max_repeat_run  最长的"同一个词连续重复"长度（退化时可达十几；正常 ≤2~3）
  distinct_ratio  去重词数 / 总词数（复读时塌到很低；正常 0.5~0.9）
  top_frac        最高频词占比
  degenerate      判据：max_repeat_run ≥ 5 或 distinct_ratio < 0.35

另含：针尖检索（needle）命中、GSM8K 命中率（答案数字是否出现）、单流 tok/s。
"""
import argparse
import json
import re
import sys
import time
import urllib.request

BASE = None
TH = False


def first_token_logits(user, k=10, timeout=900):
    """外部建议的 P0：只看 prefill 的首 token 分布。
    若每个提示的 top1 都已经是复读 token 且 margin 很大，
    ⇒ 问题在 prefill/表示/输出头之前，可直接降低 decode 侧（KV 转移、mHC delayed、CSA2）的嫌疑；
    ⇒ 反之若首 token 正常、几步后才塌，则方向完全转向 decode 状态通路。
    走 API 的 logprobs/top_logprobs，不需要服务端探针 ✓"""
    body = {"model": "/models", "messages": [{"role": "user", "content": user}],
            "chat_template_kwargs": {"thinking": TH},
            "max_tokens": 1, "temperature": 0,
            "logprobs": True, "top_logprobs": k}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    lp = (d["choices"][0].get("logprobs") or {}).get("content") or []
    tops = []
    if lp:
        for t in lp[0].get("top_logprobs", [])[:k]:
            tops.append({"tok": t.get("token"), "logprob": round(t.get("logprob", -99), 3)})
    return tops


def chat(user, n=128, timeout=1800):
    body = {"model": "/models", "messages": [{"role": "user", "content": user}],
            "chat_template_kwargs": {"thinking": TH},
            "max_tokens": n, "temperature": 0}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    dt = time.time() - t0
    txt = d["choices"][0]["message"]["content"] or ""
    u = d.get("usage", {})
    return txt, dt, u.get("completion_tokens", 0), u.get("prompt_tokens", 0)


def metrics(txt):
    """退化判据以**字符级**为主。

    教训：初版只用 `txt.split()` 分词做判据，但**中文不用空格分词**——
    一整句中文会被当成 1~4 个"词"，distinct_ratio 恒为 1.0、run 恒为 1，
    于是任何中文输出都判成"正常"。正确做法是字符级统计（对中英都成立），
    词级仅作参考。
    """
    out = {}
    # 词级（英文友好，中文仅参考）
    w = txt.split()
    if w:
        run = best = 1
        for i in range(1, len(w)):
            run = run + 1 if w[i] == w[i - 1] else 1
            best = max(best, run)
        out["w_max_repeat_run"] = best
        out["w_n_words"] = len(w)
    else:
        out["w_max_repeat_run"] = 0
        out["w_n_words"] = 0
    # 字符级（判据用）
    cs = [c for c in txt if not c.isspace()]
    if not cs:
        out.update({"max_repeat_run": 0, "distinct_ratio": 0.0, "top_frac": 1.0,
                    "degenerate": True, "n_chars": 0})
        return out
    run = best = 1
    for i in range(1, len(cs)):
        run = run + 1 if cs[i] == cs[i - 1] else 1
        best = max(best, run)
    from collections import Counter
    c = Counter(cs)
    dr = len(c) / len(cs)
    tf = c.most_common(1)[0][1] / len(cs)
    out.update({"max_repeat_run": best, "distinct_ratio": round(dr, 4),
                "top_frac": round(tf, 4), "n_chars": len(cs),
                # 阈值标定：正常中文句子 distinct≈0.6~0.9、最长同字连跑 ≤3；
                # "的"×30 这类复读 distinct≈0.03、连跑 30。两者相距一个数量级。
                "degenerate": (best >= 8 or dr < 0.15)})
    return out


FIXED = [
    ("中文解释", "请解释张量并行是如何切分权重与注意力头的，通信开销来自哪里。", 128),
    ("英文事实", "What is the capital of France? Answer in one short sentence.", 32),
    ("中文翻译", "把这句话翻译成英文：今天天气很好，我们一起去散步。", 48),
    ("简单推理", "一个班有 12 个男生和 15 个女生，老师要分成人数相等的两组，每组最多多少人？", 128),
]


def main():
    global BASE, TH
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8119)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--n-gsm8k", type=int, default=16)
    ap.add_argument("--gsm8k", default="/w/quark-int8/refs/gsm8k_sample.jsonl")
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args()

    if a.compare:
        ra = json.load(open(a.compare[0]))
        rb = json.load(open(a.compare[1]))
        for k in ("label", "degenerate_rate", "gsm8k_hit_rate", "tok_per_s", "needle_hit"):
            if k in ra or k in rb:
                print(f"{k:18s} A={ra.get(k)}   B={rb.get(k)}")
        return 0

    BASE = f"http://127.0.0.1:{a.port}"
    TH = a.thinking
    res = {"label": a.label, "thinking": TH, "items": []}

    # 1) 固定题组
    for name, p, n in FIXED:
        txt, dt, ct, pt = chat(p, n=n)
        m = metrics(txt)
        res["items"].append({"kind": "fixed", "name": name, "prompt_tokens": pt,
                             "completion_tokens": ct, "sec": round(dt, 2),
                             "text": txt[:400], **m})
        print(f"[固定] {name:8s} tok={ct:3d} {dt:5.1f}s 重复run={m['max_repeat_run']:2d} "
              f"去重={m['distinct_ratio']:.3f} {'◆退化◆' if m['degenerate'] else 'ok'}")
        print(f"        {txt[:120]!r}")

    # 2) 针尖检索
    code = "ZEPHYR-7781"
    filler = ("The quarterly report covers logistics, staffing, and procurement. " * 40)
    p = f"阅读下面的材料，然后只回答访问码是什么。\n\n{filler}\n访问码是 {code}。\n\n问题：访问码是什么？"
    txt, dt, ct, pt = chat(p, n=32)
    hit = code in txt
    res["items"].append({"kind": "needle", "name": "needle", "prompt_tokens": pt,
                         "completion_tokens": ct, "sec": round(dt, 2),
                         "text": txt[:400], "hit": hit, **metrics(txt)})
    res["needle_hit"] = hit
    print(f"[针尖] prompt={pt} tok 命中={hit}   {txt[:80]!r}")

    # 3) GSM8K
    hits = 0
    n_ok = 0
    try:
        lines = open(a.gsm8k, encoding="utf-8").read().splitlines()
    except FileNotFoundError:
        lines = []
        print(f"❌ 找不到 GSM8K 样本 {a.gsm8k} —— **不要**把本轮的 gsm8k_hit_rate=None 当成"
              f"「测过了」，请用 --gsm8k 传正确路径（宿主跑时是 $REPO/refs/...，不是 /w/...）",
              file=sys.stderr)
    if a.n_gsm8k > 0 and not lines:
        print("❌ GSM8K 题集为空 ⇒ 本轮命中率无意义", file=sys.stderr)
    for ln in lines[: a.n_gsm8k]:
        d = json.loads(ln)
        ans = d["doc"]["answer"]
        mnum = re.findall(r"####\s*([-\d.,]+)", ans)
        if not mnum:
            continue
        gold = mnum[-1].replace(",", "").rstrip(".")
        q = d["doc"]["question"].strip()
        txt, dt, ct, pt = chat(q + "\n请一步步推理，最后单独一行写「答案是 <数>」。", n=256)
        got = gold in txt.replace(",", "")
        hits += int(got)
        n_ok += 1
        res["items"].append({"kind": "gsm8k", "gold": gold, "hit": got,
                             "completion_tokens": ct, "sec": round(dt, 2),
                             "text": txt[:300], **metrics(txt)})
    res["gsm8k_hit_rate"] = round(hits / n_ok, 4) if n_ok else None
    res["gsm8k_n"] = n_ok

    # P0：首 token top-k logits（失败不影响主评测）
    res["first_token"] = {}
    for name, p_, _n in FIXED[:3]:
        try:
            tops = first_token_logits(p_)
            res["first_token"][name] = tops
            t1 = tops[0]["tok"] if tops else None
            m2 = (tops[0]["logprob"] - tops[1]["logprob"]) if len(tops) > 1 else None
            print(f"[首token] {name:8s} top1={t1!r}  margin={m2}  top5={[t['tok'] for t in tops[:5]]}")
        except Exception as e:
            print(f"[首token] {name} 失败: {type(e).__name__} {e}")
            res["first_token"][name] = {"error": str(e)}

    items = res["items"]
    res["degenerate_rate"] = round(sum(i["degenerate"] for i in items) / len(items), 4)
    tot_t = sum(i["sec"] for i in items)
    tot_c = sum(i["completion_tokens"] for i in items)
    res["tok_per_s"] = round(tot_c / tot_t, 3) if tot_t else None
    print(f"\n=== {a.label}：退化率 {res['degenerate_rate']:.2%}（{len(items)} 项）"
          f"  GSM8K 命中 {res['gsm8k_hit_rate']}（n={n_ok}）  单流 {res['tok_per_s']} tok/s"
          f"  针尖 {'命中' if hit else '未命中'} ===")
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"已写 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
