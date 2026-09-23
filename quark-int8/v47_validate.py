#!/usr/bin/env python3
"""⑫ mhc 修复后的质量/性能验收（v47）。
判据：① 常识/算术/代码补全三类 prompt 输出成句（不再是标点退化）
      ② 同请求两次 greedy 输出逐字节相同（确定性）
      ③ 长上下文（~2.5K token 大海捞针）能捞出答案 ⇒ 压缩/DSA 路径也工作
      ④ 单流 TPS（enforce_eager 下的下界）
"""
import json, sys, time, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8119"
BASE = f"http://127.0.0.1:{PORT}"

def post(path, payload, timeout=900):
    r = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())

def complete(prompt, n=32, t=0.0):
    d = post("/v1/completions", {"model": "/models", "prompt": prompt,
                                 "max_tokens": n, "temperature": t})
    return d["choices"][0]["text"], d.get("usage", {})

print("=" * 78)
print("① 三类 prompt 的补全质量（对照：修复前全是 ' ( ( ( (' 标点退化）")
print("=" * 78)
CASES = [
    ("常识", "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is"),
    ("算术", "1+1=2, 2+2=4, 3+3="),
    ("代码", "def add(a, b):\n    return"),
    ("中文", "北京是中国的首都，东京是日本的首都，巴黎是"),
]
texts = []
for name, p in CASES:
    txt, u = complete(p, 32)
    texts.append(txt)
    print(f"  [{name}] {p[:44]!r}")
    print(f"        -> {txt!r}")

print()
print("=" * 78)
print("② 确定性（同请求 greedy 两次必须逐字节相同）")
print("=" * 78)
a, _ = complete(CASES[0][1], 32)
b, _ = complete(CASES[0][1], 32)
print(f"  两次输出相同: {'✅ 是' if a == b else '❌ 否'}")
if a != b:
    print(f"    A={a!r}\n    B={b!r}")

print()
print("=" * 78)
print("③ 长上下文大海捞针（~2.5K token，越过 sliding_window=128，必须走压缩/DSA）")
print("=" * 78)
filler = ("The history of the region is long and complicated. " * 40)
needle = " The secret passcode is ZEPHYR-7781. "
prompt = (filler + needle + filler)[:9000]
prompt = "Read the text and answer with only the passcode.\n" + prompt + "\nQ: What is the secret passcode?\nA:"
try:
    txt, u = complete(prompt, 24)
    pt = u.get("prompt_tokens")
    hit = "ZEPHYR" in txt.upper() or "7781" in txt
    print(f"  prompt_tokens={pt}  输出={txt!r}")
    print(f"  捞到针: {'✅ 是' if hit else '❌ 否'}")
except Exception as e:
    print(f"  失败: {type(e).__name__}: {str(e)[:200]}")

print()
print("=" * 78)
print("④ 单流 TPS（greedy 256 token）")
print("=" * 78)
try:
    t0 = time.time()
    _, u = complete("Write a detailed explanation of how a transformer works:", 256)
    dt = time.time() - t0
    ct = u.get("completion_tokens", 0)
    print(f"  {ct} token / {dt:.1f}s = {ct/dt:.2f} tok/s（含 TTFT）")
except Exception as e:
    print(f"  失败: {e}")

print()
print("=" * 78)
print("判据汇总：① 成句  ② 确定  ③ 捞到针  ④ TPS")
print("=" * 78)
