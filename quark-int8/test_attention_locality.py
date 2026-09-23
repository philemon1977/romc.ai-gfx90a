#!/usr/bin/env python3
"""注意力"是否只用最近几个 token"的无参考判据。

原理：同一条尾巴 Q，前面接不同长度的无关前缀。若模型只能看见最近若干个 token，
则不同前缀下的"下一 token 分布"几乎相同（分布距离≈0）；正常的模型会因上下文
不同而明显改变（前缀里出现的实体/话题会改变先验）。

对照锚：把 Q 前面的无关前缀换成**相关**前缀（另一个国家的首都句），
正常的模型分布应显著不同 ⇒ 用"无关前缀 vs 相关前缀"两组距离做标尺。
"""
import json, math, sys, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8119"
BASE = f"http://127.0.0.1:{PORT}"

def dist(prompt, k=20):
    r = urllib.request.Request(BASE + "/v1/completions",
        data=json.dumps({"model": "/models", "prompt": prompt, "max_tokens": 1,
                         "temperature": 0, "logprobs": k}).encode(),
        headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(r, timeout=600).read())
    e = d["choices"][0]["logprobs"]["top_logprobs"][0]
    return {t: v for t, v in e.items()}

def jsd(p, q):
    """对称化的 KL 距离（只用两者的 top-k，缺失项按 logprob 下限补）。"""
    keys = set(p) | set(q)
    lo = min(list(p.values()) + list(q.values())) - 2.0
    P = {k: math.exp(p.get(k, lo)) for k in keys}
    Q = {k: math.exp(q.get(k, lo)) for k in keys}
    sp, sq = sum(P.values()), sum(Q.values())
    P = {k: v / sp for k, v in P.items()}; Q = {k: v / sq for k, v in Q.items()}
    m = {k: 0.5 * (P[k] + Q[k]) for k in keys}
    def kl(a, b):
        return sum(v * math.log(v / b[k]) for k, v in a.items() if v > 0)
    return 0.5 * kl(P, m) + 0.5 * kl(Q, m)

Q = "The capital of Italy is"
JUNK = ("Bananas are yellow. Rivers flow downhill. The meeting is on Tuesday. "
        "Socks come in pairs. Paint dries slowly. ")
REL = "The capital of France is Paris. The capital of Japan is Tokyo. "

cases = {
    "tail only            ": Q,
    "junk prefix  ~50 tok ": (JUNK * 3).strip() + " " + Q,
    "junk prefix ~200 tok ": (JUNK * 12).strip() + " " + Q,
    "junk prefix ~800 tok ": (JUNK * 48).strip() + " " + Q,
    "relevant prefix      ": REL + Q,
}
ds = {}
for name, p in cases.items():
    d = dist(p)
    ds[name] = d
    top = sorted(d.items(), key=lambda kv: -kv[1])[:4]
    print(f"  {name}: " + " | ".join(f"{k!r}:{v:.2f}" for k, v in top))

base = ds["tail only            "]
print()
print("=== 与 '只有尾巴' 的分布距离（JSD）===")
for name, d in ds.items():
    print(f"  {name} JSD={jsd(base, d):.4f}")
print()
print("判据：若 'junk prefix ~800 tok' 的 JSD 与 'tail only' 一样接近 0（<0.01），")
print("      说明几百个 token 的前缀对输出毫无影响 ⇒ 注意力只看最近若干 token（局部化）。")
print("      而 'relevant prefix' 应当给出明显更大的 JSD（作为尺子）。")
