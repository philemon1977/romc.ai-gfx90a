#!/usr/bin/env python3
"""长上下文单流口径：把一次请求拆成 prefill(TTFT) 与 decode，并给出 step ms / 接受率。

为什么需要它：attention backend（ROCM_ATTN vs TRITON_ATTN）在 256 token 的短上下文里
几乎不影响 decode（KV 太小），差异只在**长 KV** 上放大；同理 --max-num-batched-tokens
只影响 prefill 分块粒度 ⇒ 只看短 prompt 会得出"两者没差"的错误结论。

用法: longctx_probe.py PORT TAG [PROMPT_TOKENS] [MAXTOK] [SPEC]
"""
import json
import re
import statistics
import sys
import time
import urllib.request

PORT = sys.argv[1]
TAG = sys.argv[2]
TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 16000
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 32
SPEC = int(sys.argv[5]) if len(sys.argv) > 5 else 5

BASE = ("The following paragraph is filler used to reach a target context length. "
        "Memory bandwidth, expert routing and recurrent state updates all interact "
        "during autoregressive decoding of a mixture-of-experts language model. ")


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def build_prompt(target: int) -> str:
    """按 /tokenize 的真实 token 数拼到目标长度（重复但带序号，避免被前缀缓存整块命中）。"""
    txt = ""
    i = 0
    while True:
        i += 1
        txt += f"[{i}] " + BASE
        if i % 40 == 0:
            n = post("/tokenize", {"model": "ornith", "prompt": txt})["count"]
            if n >= target:
                return txt


def metrics():
    raw = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=60).read().decode()
    out = {}

    def g(name, typ="sum"):
        m = re.search(rf"^vllm:{name}\{{[^}}]*\}}\s+([0-9.eE+]+)$", raw, re.M)
        return float(m.group(1)) if m else 0.0

    out["ttft_sum"] = g("time_to_first_token_seconds_sum")
    out["ttft_cnt"] = g("time_to_first_token_seconds_count")
    out["ptok"] = g("prompt_tokens_total")
    out["gtok"] = g("generation_tokens_total")
    out["draft"] = g("spec_decode_num_draft_tokens_total")
    out["accept"] = g("spec_decode_num_accepted_tokens_total")
    if "ttft_sum" not in out or out["ttft_sum"] == 0.0:
        # 退路：有些版本只给 histogram bucket，取 +Inf 桶
        m = re.search(r'^vllm:time_to_first_token_seconds_bucket\{[^}]*le="\+Inf"\}\s+([0-9.eE+]+)$',
                      raw, re.M)
        out["ttft_cnt"] = float(m.group(1)) if m else 0.0
    return out


def one(prompt: str):
    m0 = metrics()
    t0 = time.perf_counter()
    d = post("/v1/completions", {"model": "ornith", "prompt": prompt, "max_tokens": MAXTOK,
                                 "temperature": 0, "ignore_eos": True})
    wall = time.perf_counter() - t0
    m1 = metrics()
    n = d["usage"]["completion_tokens"]
    dttft = m1["ttft_sum"] - m0["ttft_sum"]
    dcnt = m1["ttft_cnt"] - m0["ttft_cnt"]
    ttft = dttft / dcnt if dcnt else float("nan")
    dd = m1["draft"] - m0["draft"]
    da = m1["accept"] - m0["accept"]
    # SPEC=0 时没有草稿 ⇒ 一步一个 token；SPEC>0 时 steps = 草稿数/深度
    steps = dd / SPEC if SPEC else n
    dec = wall - ttft
    return dict(ptok=m1["ptok"] - m0["ptok"], ntok=n, wall=wall, ttft=ttft,
                dec=dec, steps=steps, step_ms=1000 * dec / steps if steps else float("nan"),
                tokperstep=n / steps if steps else float("nan"),
                acc=100 * da / dd if dd else float("nan"), tps=n / dec if dec > 0 else float("nan"))


def main():
    prompt = build_prompt(TARGET)
    real = post("/tokenize", {"model": "ornith", "prompt": prompt})["count"]
    print(f"[{TAG}] prompt={real} tokens, max_tokens={MAXTOK}, SPEC={SPEC}", flush=True)
    one(prompt)  # warmup（首个请求丢弃）
    rows = [one(prompt) for _ in range(3)]
    for i, r in enumerate(rows, 1):
        print(f"  rep{i}: ptok {r['ptok']:.0f} | TTFT {r['ttft']:.2f}s "
              f"(prefill {r['ptok']/r['ttft']:.0f} tok/s) | decode {r['ntok']} tok in "
              f"{r['dec']:.2f}s = {r['tps']:.2f} tok/s | steps {r['steps']:.0f} "
              f"| step {r['step_ms']:.1f} ms | {r['tokperstep']:.2f} tok/step "
              f"(accept {r['acc']:.1f}%)", flush=True)
    med = lambda k: statistics.median([r[k] for r in rows])
    print(f"[{TAG}] MEDIAN @{real}tok ctx: TTFT {med('ttft'):.2f}s "
          f"(prefill {med('ptok')/med('ttft'):.0f} tok/s) | decode {med('tps'):.2f} tok/s "
          f"| step {med('step_ms'):.1f} ms | accept {med('acc'):.1f}%")


if __name__ == "__main__":
    main()
