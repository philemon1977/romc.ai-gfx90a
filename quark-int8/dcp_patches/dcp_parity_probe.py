#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCP 等价性（parity）探针 —— ①③ 的正式判据（2026-09-20 评审后取代"针尖命中"）。

为什么不用针尖判 DCP：GLM 的 DSA 只对 top-2048 做注意力，1M 下命中与否主要由选择决定，
与 DCP 的 LSE 合并是否正确无关。DCP 写错的表现是**同一 prompt 下输出与 DCP=1 不同**。

用法（同一 prompt、同一权重，跑两次起服）：
  # DCP=1 起服后
  python3 dcp_parity_probe.py --port 8121 --out /tmp/parity_dcp1.json --tokens 4096
  # DCP=2 起服后
  python3 dcp_parity_probe.py --port 8121 --out /tmp/parity_dcp2.json --tokens 4096 \
      --compare /tmp/parity_dcp1.json
判据：贪心输出逐字节相同（或 logprob 相对误差 < 阈值）⇒ DCP 合并正确。
注意：填充必须逐条唯一（用 agent_bench2.doc），否则 top-2048 选择会共振，两次跑的选择集可能本来就不同。
"""
import argparse, importlib.util, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def load_doc():
    """复用 agent_bench2 的唯一句填充（避免两套生成器漂移）。"""
    path = os.path.join(os.path.dirname(HERE), "agent_bench2.py")
    spec = importlib.util.spec_from_file_location("agent_bench2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.doc


def run(base, model, prompt, max_tokens):
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "logprobs": 5, "stream": False,
    }).encode()
    req = urllib.request.Request(base + "/v1/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        d = json.load(r)
    dt = time.time() - t0
    ch = d["choices"][0]
    lp = ch.get("logprobs") or {}
    return {
        "text": ch.get("text", ""),
        "tokens": lp.get("tokens", []),
        "top1_logprob": [t.get("logprob") for t in (lp.get("top_logprobs") or [{}])],
        "prompt_tokens": d.get("usage", {}).get("prompt_tokens"),
        "seconds": dt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8121)
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--tokens", type=int, default=4096, help="文档目标 token（唯一句填充）")
    ap.add_argument("--answer-tokens", type=int, default=32)
    ap.add_argument("--question", default="Summarize the third record in one short sentence.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default="", help="与先前一次的结果比对（DCP=1 基线）")
    a = ap.parse_args()

    doc = load_doc()
    prompt = doc(a.tokens, tag="P") + "\n\nQuestion: " + a.question + "\nAnswer:"
    got = run("http://127.0.0.1:%d" % a.port, a.model, prompt, a.answer_tokens)
    json.dump({"tokens_target": a.tokens, **got}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("prompt=%s tokens 生成 %d token，耗时 %.1fs → %s"
          % (got["prompt_tokens"], len(got["tokens"]), got["seconds"], a.out))
    print("输出: %r" % got["text"][:200])

    if not a.compare:
        print("[baseline] 保存为基线；DCP=2 起服后用 --compare %s 复跑" % a.out)
        return
    ref = json.load(open(a.compare))
    same_text = ref["text"] == got["text"]
    same_tokens = ref["tokens"] == got["tokens"]
    print("=== parity vs %s ===" % a.compare)
    print("  文本完全一致: %s" % same_text)
    print("  token 序列一致: %s" % same_tokens)
    if not same_tokens:
        n = min(len(ref["tokens"]), len(got["tokens"]))
        diverge = next((i for i in range(n) if ref["tokens"][i] != got["tokens"][i]), n)
        print("  首个分歧位置: %d（前 %d 个 token 一致）" % (diverge, diverge))
        print("  基线: %r" % ref["text"][:160])
        print("  本次: %r" % got["text"][:160])
        print("⇒ 不一致：先查 B 的 index 切分与 C 的合并（见 dcp_patches/README.md 尺子顺序）")
    else:
        print("⇒ DCP parity 通过（贪心逐 token 相同）")


if __name__ == "__main__":
    main()
