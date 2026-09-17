#!/usr/bin/env python3
"""decode 速度解剖：把"t/s"拆成 **真 token 数 / 步数 / step ms / MTP 接受率**，并核对
多轮探针用 SSE 分块数当 token 数是否低估。

三种 workload 对比（同一台服务，同一 SPEC=5）：
  count   高度可预测（"从一数到九十"），25 token 上下文
  explain 自由文本解释，~30 token 上下文
  vibecode 多轮探针那种：~2.4k 上下文 + 1500 token 自由发挥
用法：decode_anatomy.py PORT
"""
import json
import re
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8117"
SPEC = 5

COUNT = "Count slowly from one to ninety, writing each number in words on its own line."
EXPLAIN = ("Explain in detail why int4 quantization reduces memory bandwidth pressure "
           "during decoding, covering weights and KV cache.")
BODY = ("The following is a repository snapshot. Read it carefully before answering. "
        "File launcher.sh sets ROCM_PATH, MODEL_PATH and GPU_MEM_UTIL, then execs vllm. "
        "File kernel.py registers a gate GEMV kernel under flag MI250_GATE_GEMV. "
        "File ledger.jsonl records every experiment verdict with a doc pointer. ") * 30
VIBE = BODY + ("\nUser: In this repository, the launcher scripts live under models/ and the patches "
               "under patches/gfx90a/. Explain what the file mi250_kernels.py does, step by step, and "
               "list the environment flags it defines.\n")


def scrape():
    txt = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=60).read().decode()
    out = {}
    for k in ("generation_tokens", "prompt_tokens", "spec_decode_num_draft_tokens",
              "spec_decode_num_accepted_tokens"):
        m = re.search(rf"^vllm:{k}_total\{{[^}}]*\}}\s+([0-9.eE+]+)$", txt, re.M)
        out[k] = float(m.group(1)) if m else 0.0
    return out


def gen(prompt, maxtok):
    body = json.dumps({"model": "ornith", "prompt": prompt, "max_tokens": maxtok,
                       "temperature": 0, "ignore_eos": True, "stream": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                if ttft is None:
                    ttft = time.time() - t0
                try:
                    if json.loads(line[6:])["choices"][0].get("text"):
                        chunks += 1
                except Exception:
                    pass
    return ttft, time.time() - t0, chunks


def run(tag, prompt, maxtok, warm=True):
    if warm:
        gen(prompt[:200], 8)          # 预热（不复用同一 prompt 的缓存，故不污染命中统计）
    c0 = scrape()
    ttft, dt, chunks = gen(prompt, maxtok)
    c1 = scrape()
    dg = c1["generation_tokens"] - c0["generation_tokens"]
    dd = c1["spec_decode_num_draft_tokens"] - c0["spec_decode_num_draft_tokens"]
    da = c1["spec_decode_num_accepted_tokens"] - c0["spec_decode_num_accepted_tokens"]
    dp = c1["prompt_tokens"] - c0["prompt_tokens"]
    dec = max(dt - (ttft or 0), 1e-3)
    steps = dd / SPEC if dd else float("nan")
    print(f"[{tag}] prompt {int(dp)} tok | 生成 {int(dg)} tok"
          f"{'（SSE 分块 ' + str(chunks) + '）' if chunks != dg else '（SSE 分块与真值一致）'}")
    print(f"       TTFT {ttft:.2f}s 总 {dt:.2f}s ⇒ decode **{dg/dec:.2f} t/s**"
          f" | 步数 {steps:.0f} | {dg/steps if steps==steps else float('nan'):.2f} tok/步"
          f" | 接受率 {100*da/dd if dd else 0:.1f}% | step {(1000*dec/steps) if steps==steps else float('nan'):.1f} ms")
    # 自洽性核对：generation ≈ steps + accepted
    if dd:
        pred = steps + da
        print(f"       自洽性：steps+accepted = {pred:.0f} vs 生成 {int(dg)}"
              f" ⇒ {'一致 ✅' if abs(pred-dg) <= max(3, 0.03*dg) else '不一致 ⚠️（计数器口径）'}")
    return dg / dec


def main():
    print(f"=== decode 解剖（PORT={PORT}, SPEC={SPEC}）===")
    run("count-256", COUNT, 256)
    run("explain-256", EXPLAIN, 256)
    run("vibecode-1500@2.4k", VIBE, 1500)
    # 再压一次长上下文：把同一段正文加长到 ~7k，模拟第 4 轮
    long = (BODY * 3) + VIBE[-600:]
    run("vibecode-1500@7k", long, 1500)


if __name__ == "__main__":
    main()
