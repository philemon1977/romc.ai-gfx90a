#!/usr/bin/env python3
"""前缀缓存 × MTP 的三臂探针：hits 到底是不是 0，以及 SPEC=0 会不会恢复。

每臂：
  1) 发两次**同一个长 prompt**（~5k tokens，流式），量 TTFT(冷) 与 TTFT(暖)
  2) 读 /metrics 的 vllm:prefix_cache_{queries,hits}_total 前后差值
用法：prefix_probe.py PORT TAG
"""
import json
import re
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8117"
TAG = sys.argv[2] if len(sys.argv) > 2 else "probe"

PARA = ("The AMD Instinct MI250X is a dual-die accelerator based on the CDNA 2 architecture. "
        "Quantization lowers memory footprint and memory traffic during autoregressive decoding. "
        "For a mixture-of-experts model, expert weights dominate the parameter count, so converting "
        "only the routed experts to INT8 halves the model size with limited accuracy loss. ")


def long_prompt(reps=40):
    return (PARA * reps) + "\nReply with the single word OK."


def scrape():
    txt = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=60).read().decode()
    out = {}
    for key in ("prefix_cache_queries", "prefix_cache_hits"):
        m = re.search(rf"^vllm:{key}_total\{{[^}}]*\}}\s+([0-9.eE+]+)$", txt, re.M)
        out[key] = float(m.group(1)) if m else None
    return out


def timed(prompt, maxtok=8):
    body = json.dumps({"model": "ornith", "prompt": prompt, "max_tokens": maxtok,
                       "temperature": 0, "ignore_eos": True, "stream": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
    return ttft, time.time() - t0, n


def main():
    p = long_prompt()
    # 用字符数粗估 token 数（英文 ~4 字符/token）
    print(f"[{TAG}] prompt ≈ {len(p)} 字符 ≈ {len(p)//4} tokens")
    c0 = scrape()
    ttft1, tot1, _ = timed(p)
    c1 = scrape()
    ttft2, tot2, _ = timed(p)
    c2 = scrape()
    print(f"[{TAG}] 冷请求: TTFT {ttft1:.2f}s 总 {tot1:.2f}s")
    print(f"[{TAG}] 暖请求: TTFT {ttft2:.2f}s 总 {tot2:.2f}s   ⇒ 加速 {ttft1/max(ttft2,1e-3):.1f}×")
    for name, a, b in (("queries", c0["prefix_cache_queries"], c2["prefix_cache_queries"]),
                       ("hits", c0["prefix_cache_hits"], c2["prefix_cache_hits"])):
        if a is None or b is None:
            print(f"[{TAG}] {name}: 指标不可用")
        else:
            print(f"[{TAG}] {name}: +{b-a:.0f}  (总 {b:.0f})")
    q = (c2["prefix_cache_queries"] or 0) - (c0["prefix_cache_queries"] or 0)
    h = (c2["prefix_cache_hits"] or 0) - (c0["prefix_cache_hits"] or 0)
    if q:
        print(f"[{TAG}] ⇒ 命中率 {100*h/q:.1f}%（hits/queries）")
    print(f"[{TAG}] 注：queries/hits 的中间一次（暖请求前）: "
          f"{c1['prefix_cache_queries']}, {c1['prefix_cache_hits']}")


if __name__ == "__main__":
    main()
