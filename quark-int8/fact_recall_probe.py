#!/usr/bin/env python3
# 事实召回探针（单变量：镜像版本 09-14→09-18 后复读吸引子是否消失）
# 判据出处：2026-09-19 10:34 诊断记录——旧镜像上裸文本 top1 依次为
#   '\n'(应为北京) / ' the'(应为Paris) / '，'(应为北京) / ''(应为数字)。
# 2026-09-20 改进：加"期望子串"表 ⇒ 判定机器化、跨臂可比（此前靠肉眼看输出）。
import json, urllib.request, sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8119
MODEL = sys.argv[2] if len(sys.argv) > 2 else "/models"
# (prompt, 期望出现的子串元组, 判据说明)
PROMPTS = [
    ("中国的首都是", ("北京",), "首都=北京"),
    ("The capital of France is", ("Paris",), "首都=Paris"),
    ("法国首都巴黎，日本首都东京，中国首都", ("北京",), "接龙=北京"),
    ("1+1=", ("2",), "算术=2"),
    ("水的化学式是", ("H2O", "H₂O"), "化学式"),
    ("DeepSeek 是由哪家", ("公司",), "出处=公司"),
]

def one(p):
    body = json.dumps({"model": MODEL, "prompt": p, "max_tokens": 8,
                       "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["text"]

if __name__ == "__main__":
    bad = 0
    ok = 0
    n = 0
    for p, want, why in PROMPTS:
        try:
            txt = one(p)
            hit = any(w in txt for w in want)
            n += 1
            ok += int(hit)
            print(f"{p!r}  ->  {txt!r}   [{'PASS' if hit else 'FAIL'} {why}]", flush=True)
        except Exception as e:
            print(f"{p!r}  ->  <ERR {e}>", flush=True)
            bad += 1
    if n:
        print(f"=== 事实召回 {ok}/{n} = {100.0*ok/n:.1f}%（请求失败 {bad}） ===", flush=True)
    sys.exit(1 if (bad or (n and ok < n)) else 0)
