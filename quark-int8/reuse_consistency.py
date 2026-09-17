#!/usr/bin/env python3
"""数值硬门 V2：**命中复用** 与 **冷算** 必须给出逐步相同的输出与 logprobs。

为什么必须做：递归状态（SSM/conv）复用若读错块、或在错误的边界写快照，**不会报任何错**，
只会让输出悄悄变差。所以"开前缀缓存后 warm 请求 vs cold 请求"的输出一致性，是唯一能抓住
"静默算错"的测试。

做法：同一个长 prompt（>2 块，且尾部故意不落在 block 边界）连发 3 次：
  run1 = 冷（无命中，全量算）   run2/run3 = 暖（应命中前缀缓存）
判据：三次的 **token id 序列完全相同**，且 **每个 token 的 logprob 最大差 ≤ TOL**。
（⚠️ 用**生成 token 的 logprobs**，不要用 prompt_logprobs —— SPEC=5 下后者被污染，本会话已实撞。）

用法：reuse_consistency.py PORT TAG [MAxTOK=64] [TOL=1e-2]
"""
import json
import re
import sys
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8127"
TAG = sys.argv[2] if len(sys.argv) > 2 else "rc"
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 64
TOL = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-2

# 稳定、可复现的长 prompt（≈3k tokens），尾部不落在 544 的整数倍上
BODY = ("The following is a repository snapshot. Read it carefully before answering. "
        "File launcher.sh sets ROCM_PATH, MODEL_PATH and GPU_MEM_UTIL, then execs vllm. "
        "File kernel.py registers a gate GEMV kernel under flag MI250_GATE_GEMV. "
        "File ledger.jsonl records every experiment verdict with a doc pointer. ") * 55
TAIL = ("\nQuestion: in one short sentence, what does MI250_GATE_GEMV change, and why "
        "must the other kernel flags stay off? Answer now.")


def scrape():
    txt = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=60).read().decode()
    out = {}
    for k in ("prefix_cache_queries", "prefix_cache_hits"):
        m = re.search(rf"^vllm:{k}_total\{{[^}}]*\}}\s+([0-9.eE+]+)$", txt, re.M)
        out[k] = float(m.group(1)) if m else 0.0
    return out


def gen(prompt):
    body = json.dumps({"model": "ornith", "prompt": prompt, "max_tokens": MAXTOK,
                       "temperature": 0, "ignore_eos": True,
                       "logprobs": 1}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    ch = d["choices"][0]
    lp = ch.get("logprobs") or {}
    toks = lp.get("tokens") or []
    lps = lp.get("token_logprobs") or []
    return ch.get("text", ""), toks, lps


def main():
    p = BODY + TAIL
    print(f"[{TAG}] prompt ≈ {len(p)//4} tok，max_tokens={MAXTOK}，容差 {TOL}")
    runs = []
    for i in range(1, 4):
        c0 = scrape()
        text, toks, lps = gen(p)
        c1 = scrape()
        dh = c1["prefix_cache_hits"] - c0["prefix_cache_hits"]
        dq = c1["prefix_cache_queries"] - c0["prefix_cache_queries"]
        runs.append((toks, lps))
        print(f"[{TAG}] run{i}: hits +{int(dh)}/{int(dq)}  logprobs {len(lps)} 个  "
              f"首 token {toks[0] if toks else '?'!r}", flush=True)

    base_toks, base_lps = runs[0]
    ok_tok = True
    worst = 0.0
    for i, (toks, lps) in enumerate(runs[1:], start=2):
        same = toks == base_toks
        ok_tok &= same
        d = 0.0
        for a, b in zip(base_lps, lps):
            if a is None or b is None:
                continue
            d = max(d, abs(a - b))
        worst = max(worst, d)
        print(f"[{TAG}] run1 vs run{i}: token 序列{'相同 ✅' if same else '不同 ❌'}"
              f"  最大 |Δlogprob| = {d:.3e}")
    verdict = "PASS ✅" if (ok_tok and worst <= TOL) else "FAIL ❌"
    print(f"[{TAG}] 数值硬门 V2：{verdict}（判据：token 序列逐位相同 且 |Δlogprob| ≤ {TOL}；"
          f"实测最差 {worst:.3e}）")
    return 0 if verdict.startswith("PASS") else 2


if __name__ == "__main__":
    sys.exit(main())
