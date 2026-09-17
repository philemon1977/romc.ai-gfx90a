#!/usr/bin/env python3
"""Locate why the quality probe reports NLL ~12.9 (≈ ln(vocab), i.e. uniform-random logprobs).

Two independent vLLM paths score the same text:
  A) prompt_logprobs=0            <- what nll_probe.py uses
  B) echo=True + logprobs=1       <- the classic path; also returns the token ids for an
                                     alignment check (does the returned token sequence match
                                     the text we asked to score?)
If A is ~12.9 while B is ~2.0, the model is fine and path A is broken.

usage: diag_nll.py PORT MODEL LABEL
"""
import json
import math
import sys
import urllib.request

sys.path.insert(0, "/home/qiba/ROCm.AI/quark-int8")
from nll_probe import TEXT  # same sample text

port, model, label = sys.argv[1], sys.argv[2], sys.argv[3]


def post(payload):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


print(f"=== [{label}] 路径 A: prompt_logprobs=0（nll_probe.py 用的）===")
d = post({"model": model, "prompt": TEXT, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0})
pl = d["choices"][0].get("prompt_logprobs")
if pl:
    lps = [next(iter(e.values()))["logprob"] for e in pl[1:] if e]
    nll = -sum(lps) / len(lps)
    print(f"  NLL={nll:.4f} (ppl {math.exp(nll):.2f})  tokens={len(lps)}")
    print(f"  前 5 个 logprob: {[round(x,2) for x in lps[:5]]}   （若都 ≈ -12 附近 ⇒ 随机）")
else:
    print("  prompt_logprobs 不可用")

print(f"\n=== [{label}] 路径 B: echo=True + logprobs=1 ===")
d2 = post({"model": model, "prompt": TEXT, "max_tokens": 0, "temperature": 0,
           "echo": True, "logprobs": 1})
ch = d2["choices"][0]
toks = ch.get("logprobs", {}).get("tokens")
tlp = ch.get("logprobs", {}).get("token_logprobs")
if toks:
    # 前若干项是 prompt 的 token（echo 会把 prompt 也返回）
    print(f"  返回 token 数={len(toks)}")
    print(f"  前 12 个 token: {toks[:12]}")
else:
    print("  echo 路径未返回 tokens")
if tlp:
    vals = [x for x in tlp if x is not None]
    # 第一个 token 的 logprob 通常为 None（无前文）
    prompt_vals = vals[:-1] if len(vals) > 1 else vals
    nll_b = -sum(prompt_vals) / len(prompt_vals)
    print(f"  NLL={nll_b:.4f} (ppl {math.exp(nll_b):.2f})  scored={len(prompt_vals)}")
    print(f"  前 5 个 logprob: {[round(x,2) for x in prompt_vals[:5]]}")
    print(f"\n  ⇒ 两条路径差异: A={nll:.4f}  B={nll_b:.4f}" if pl else "")
