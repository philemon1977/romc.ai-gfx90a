#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash **正确用法**下的评测（这才是官方支持的形态）。

为什么单独写：checkpoint 目录里**没有 chat_template**，vLLM 的 `deepseek_v41`
tokenizer 自己实现了一份（内部调用从 `encoding/encoding.py` 移植来的 `encode_messages`），
**默认 thinking=True**；而官方 `inference/generate.py` 的默认是 `thinking_mode="chat"`，
且**根本没有"裸文本续写"这条路径**。
⇒ 用 `/v1/completions` 喂裸文本得到的"退化"输出**不能**作为模型好坏的证据。

用法:
  python3 eval_dsv41_chat.py [port] [--thinking] [--needle-tokens N]
"""
import argparse, json, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("port", nargs="?", default="8119")
ap.add_argument("--thinking", action="store_true", help="用 thinking 模式（vLLM 的默认）")
ap.add_argument("--needle-tokens", type=int, default=0)
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"
TH = bool(a.thinking)

def chat(user, n=64, timeout=1800):
    payload = {"model": "/models", "messages": [{"role": "user", "content": user}],
               "max_tokens": n, "temperature": 0,
               "chat_template_kwargs": {"thinking": TH}}
    r = urllib.request.Request(BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(r, timeout=timeout).read())
    return d, time.time() - t0

CASES = [
    ("算术", "What is 2+2? Answer with just the number.", ["4"]),
    ("常识-意大利", "What is the capital of Italy? Answer in one word.", ["Rome"]),
    ("常识-日本", "What is the capital of Japan? Answer in one word.", ["Tokyo"]),
    ("事实", "Who was the first president of the United States? Answer with just the name.", ["Washington"]),
    ("中文", "北京是中国的首都。请用一句话说出法国的首都是哪里。", ["巴黎"]),
    ("代码", "Write a Python one-liner that returns the sum of a list `a`. Answer with code only.", ["sum("]),
]
print(f"### thinking={TH}  （chat_template_kwargs）")
npass = 0
for name, q, keys in CASES:
    try:
        d, dt = chat(q, n=96 if not TH else 384)
        m = d["choices"][0]["message"]
        c = (m.get("content") or "").strip()
        ok = any(k.lower() in c.lower() for k in keys)
        npass += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {c[:120]!r}  ({dt:.1f}s)")
        if m.get("reasoning"):
            print(f"        reasoning: {m['reasoning'][:100]!r}")
    except Exception as e:
        print(f"[ERR ] {name}: {type(e).__name__}: {str(e)[:120]}")
print(f"### 通过 {npass}/{len(CASES)}")

if a.needle_tokens > 0:
    filler = ("The quarterly logistics report covers warehousing, freight and customs. " * 60)
    body = filler * max(1, a.needle_tokens * 4 // len(filler) + 1)
    body = body[: a.needle_tokens * 4] + "\nIMPORTANT: the vault access code is ZEPHYR-7781.\n" + filler[:400]
    q = ("Read the text below and answer with ONLY the vault access code.\n\n" + body
         + "\n\nQuestion: What is the vault access code?")
    try:
        d, dt = chat(q, n=64)
        m = d["choices"][0]["message"]
        c = (m.get("content") or "")
        print(f"[needle ~{a.needle_tokens} tok] prompt_tokens={d['usage'].get('prompt_tokens')} "
              f"{c[:150]!r}  {'✅ 捞到' if 'ZEPHYR' in c.upper() or '7781' in c else '❌ 没捞到'}  ({dt:.1f}s)")
    except Exception as e:
        print(f"[needle] 失败 {type(e).__name__}: {str(e)[:140]}")
