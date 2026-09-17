#!/usr/bin/env python3
"""多轮（Claude-Code / DSH 式）前缀复用画像：逐轮累积上下文，量每轮 TTFT / 命中 / decode。

为什么必须先量这个：
  align 模式的状态快照只在"某个 scheduler step 的末 token 且块对齐"时产生 ⇒
  **prefill 的块尾有快照，而 decode 步的块尾通常没有**。真实多轮里，turn k 的 prompt 是
  turn k-1 的 prompt + 生成的回答 + 新消息，而"生成的回答"那段是 decode 出来的
  ⇒ align 能复用的部分可能只到上一轮 prompt 的末尾，**上一轮回答（编码 agent 里往往 1–5k token）
  要重算**。这正是本探针要量出的东西；也是 'all' 模式（每块边界都留快照）要赢的地方。

用法：multiturn_probe.py PORT TAG [TURNS] [ANSWER_TOK]
"""
import json
import re
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8127"
TAG = sys.argv[2] if len(sys.argv) > 2 else "mt"
TURNS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
ANS = int(sys.argv[4]) if len(sys.argv) > 4 else 160

# 仿 Claude-Code 的固定前缀（system + 工具说明），写死保证跨臂可比
SYSTEM = (
    "You are a coding agent working inside a repository. You have these tools: "
    "read_file(path), write_file(path, content), run_shell(cmd), grep(pattern, glob). "
    "Always inspect the code before editing. Keep changes minimal and consistent with the "
    "existing style. After editing, run the tests and report exactly what changed. "
    "Never fabricate file contents; if a file is unknown, read it first. "
) * 30

TASK = ("\nUser: In this repository, the launcher scripts live under models/ and the patches "
        "under patches/gfx90a/. Explain what the file mi250_kernels.py does, step by step, and "
        "list the environment flags it defines. Then answer the follow-up questions.\n")
FOLLOWUP = ("\nUser: Now, based on what you said, write the exact shell command to enable only the "
            "gate GEMV flag, and explain why the other flags must stay off.\n")


def scrape():
    txt = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=60).read().decode()
    out = {}
    for k in ("prefix_cache_queries", "prefix_cache_hits"):
        m = re.search(rf"^vllm:{k}_total\{{[^}}]*\}}\s+([0-9.eE+]+)$", txt, re.M)
        out[k] = float(m.group(1)) if m else 0.0
    return out


def call(prompt, maxtok):
    body = json.dumps({"model": "ornith", "prompt": prompt, "max_tokens": maxtok,
                       "temperature": 0, "ignore_eos": True, "stream": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    text = []
    n = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                if ttft is None:
                    ttft = time.time() - t0
                try:
                    d = json.loads(line[6:])
                    piece = d["choices"][0].get("text") or ""
                    if piece:
                        text.append(piece)
                        n += 1
                except Exception:
                    pass
    dt = time.time() - t0
    return ttft, dt, n, "".join(text)


def main():
    conv = SYSTEM + TASK
    print(f"[{TAG}] 轮数={TURNS} 每轮生成={ANS} tok；前缀≈{len(conv)//4} tok")
    print(f"[{TAG}] {'轮':>3} {'prompt tok':>11} {'命中 tok':>10} {'共享前缀':>9} {'复用率':>7} "
          f"{'TTFT s':>7} {'decode t/s':>10}")
    for turn in range(1, TURNS + 1):
        c0 = scrape()
        ttft, dt, ntok, ans = call(conv, ANS)
        c1 = scrape()
        dq = c1["prefix_cache_queries"] - c0["prefix_cache_queries"]
        dh = c1["prefix_cache_hits"] - c0["prefix_cache_hits"]
        prompt_tok = int(dq) if dq else None
        tps = ntok / max(dt - (ttft or 0), 1e-3)
        print(f"[{TAG}] {turn:>3} {prompt_tok if prompt_tok else '?':>11} {int(dh):>10} "
              f"{'':>9} {100*dh/max(dq,1):>6.1f}% {ttft:>7.2f} {tps:>10.2f}", flush=True)
        # 下一轮：把本轮回答 + 新问题接上（模拟逐条多轮、上下文累积）
        conv += ans + FOLLOWUP
    print(f"[{TAG}] 说明：'共享前缀' = 本轮 prompt 中来自上一轮的部分；命中率分母是该轮的 queries(≈prompt)")


if __name__ == "__main__":
    main()
