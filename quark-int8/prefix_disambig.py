#!/usr/bin/env python3
"""消解 mi250x-recipe-ops 里 arms[2].lessons[3] 与本会话实测的冲突。

冲突双方：
  技能 L3："'prefix cache 被 MTP 架空'（kv_cache_utils 警告）是误判，实测复用率 70.5%→94.4% 共存默认开"
  本会话  ：qwen4_exp(混合 GDN+MTP) 下**跨请求**复用改前恒 0，打标注补丁后才 87%
两者不可能同口径 ⇒ 最可能是"跨请求顺序复用"与"批内共享/同请求 chunked-prefill 自命中"被混为一谈。

本探针用三个**各自独占一份 prompt** 的组把它们分开（组间无污染）：
  G1 顺序同 prompt（跨请求）：P1 P1 P1 P1 —— 上一请求的 KV 已入池，真跨请求复用
  G2 并发同 prompt（批内共享）：4×P2 同一批发出 —— 批内共享前缀（不跨请求）
  G3 交替异内容（对照组）    ：P3 Q3 P3 Q3（Q3 只改中段）—— 应当≈0，用来证明计数器不是恒涨

三重取证：
  ① /metrics 计数增量：prefix_cache_hits_total / prefix_cache_queries_total
  ② 每请求 usage.prompt_tokens_details.cached_tokens（若该版本提供）
  ③ 引擎自己的周期行 "Prefix cache hit rate: X%"（L3 引的很可能是这个数）

用法：prefix_disambig.py PORT MODEL TAG [MAXTOK]
"""
import json
import os
import re
import sys
import threading
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8107"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-flash-next"
TAG = sys.argv[3] if len(sys.argv) > 3 else "boot"
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 8
BASE = f"http://127.0.0.1:{PORT}"


def units(n_units, salt=""):
    """构造 ~n_units*13 token 的确定性文本。

    ⚠️ 2026-09-21 修正（第一版探针的致命缺陷）：salt 必须进**每一行**，不能只放开头。
    只放开头时，各组正文逐字相同 ⇒ vLLM 的块级前缀缓存会命中**共享正文**（实测恒 12/13 块
    = 4800/5325 = 90.1%，连异内容对照组也"命中"），于是测的完全不是'同 prompt 跨请求复用'。
    这也给出 L3「70.5%→94.4%」的一个可能解释：那是**共享正文/系统提示**的部分前缀口径。
    """
    head = f"[{salt}] " if salt else ""
    body = " ".join(
        f"{salt} unit{i:04d} alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        for i in range(n_units)
    )
    return head + body


def differ(txt, tag):
    """只改中段：保证前缀不同、长度接近，用于异内容对照组。"""
    mid = len(txt) // 2
    return txt[:mid] + f" //{tag}// " + txt[mid:]


def scrape():
    txt = urllib.request.urlopen(f"{BASE}/metrics", timeout=60).read().decode()
    out = {"hits": 0.0, "queries": 0.0, "gauges": {}, "names": []}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z0-9_:]+)(\{[^}]*\})?\s+([0-9.eE+]+)$", line)
        if not m:
            continue
        name, _lbl, val = m.group(1), m.group(2), float(m.group(3))
        if name.endswith("prefix_cache_hits_total"):
            out["hits"] += val
            out["names"].append(name)
        elif name.endswith("prefix_cache_queries_total"):
            out["queries"] += val
            out["names"].append(name)
        elif "prefix_cache" in name and not name.endswith("_total"):
            out["gauges"][name] = val
    return out


def req(prompt, maxtok=None):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": maxtok or MAXTOK,
                       "temperature": 0, "ignore_eos": True}).encode()
    r = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(r, timeout=1800))
    u = d.get("usage", {}) or {}
    det = (u.get("prompt_tokens_details") or {}) if isinstance(u.get("prompt_tokens_details"), dict) else {}
    return {"dt": time.time() - t0, "prompt": u.get("prompt_tokens"), "out": u.get("completion_tokens"),
            "cached": det.get("cached_tokens")}


def seq_trace(name, prompt, n=6):
    """顺序同 prompt 的**逐请求**增量：这是分辨'第几次开始命中'的唯一判据。

    汇总口径会骗人：同一批里既有命中的也有不命中的，45% 这种数看不出模式。
    """
    req(units(8, salt=f"warm-{name}-{TAG}"))          # 预热用异内容
    prev = scrape()
    h0, q0 = prev["hits"], prev["queries"]
    print(f"  {name}（逐请求，n={n}）")
    for i in range(1, n + 1):
        t = req(prompt)
        cur = scrape()
        dq, dh = cur["queries"] - q0, cur["hits"] - h0
        print(f"    #{i}  t={t['dt']:>5.2f}s  Δq={dq:>6.0f}  Δh={dh:>6.0f}  "
              f"本次hit%={100 * dh / dq if dq else 0:>5.1f}  累计hit%={100 * cur['hits'] / cur['queries'] if cur['queries'] else 0:>5.1f}")
        h0, q0 = cur["hits"], cur["queries"]
    return None


def group(name, prompts, concurrent=False):
    """prompts: list[str]（顺序组逐个发；并发组同批发出）"""
    # 预热：用**另一份**内容，避免污染本组
    req(units(8, salt=f"warm-{name}-{TAG}"))
    c0 = scrape()
    rows = []
    if concurrent:
        res = [None] * len(prompts)
        def w(i):
            res[i] = req(prompts[i])
        ts = [threading.Thread(target=w, args=(i,)) for i in range(len(prompts))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        rows = res
    else:
        for p in prompts:
            rows.append(req(p))
    c1 = scrape()
    dq, dh = c1["queries"] - c0["queries"], c1["hits"] - c0["hits"]
    cached = [r.get("cached") for r in rows]
    print(f"  {name:<26} Δqueries={dq:>7.0f} Δhits={dh:>7.0f} hit%={100 * dh / dq if dq else 0:>6.1f}  "
          f"cached_tokens/req={cached}  prompt_tok={[r.get('prompt') for r in rows]}  "
          f"out_tok={[r.get('out') for r in rows]}  t={[round(r['dt'], 2) for r in rows]}")
    return {"group": name, "dqueries": dq, "dhits": dh, "cached": cached}


# ⚠️ 2026-09-21 修正 2：标签必须**每次运行唯一**（含 TAG），否则上一轮留在 KV 池里的块会让
# "全新"请求也报高命中（实测 93.1% 恒定），于是整场测量作废。TAG 由命令行给。
P1 = units(280, salt=f"{TAG}-G1a")
P2 = units(280, salt=f"{TAG}-G2a")
P3 = units(280, salt=f"{TAG}-G3a")
Q3 = differ(P3, f"{TAG}-CTRLb")

print(f"[{TAG}] PORT={PORT} MODEL={MODEL} maxtok={MAXTOK}  prompt≈{len(P1.split())} words")
s = scrape()
print(f"  计数器名: {sorted(set(s['names']))}  其它 gauge: {list(s['gauges'].keys())[:6]}")
print(f"  起始累计: hits={s['hits']:.0f} queries={s['queries']:.0f}")
res = []
res.append(group("G1 顺序同 prompt(跨请求)", [P1] * 4))
res.append(group("G2 并发同 prompt(批内)", [P2] * 4, concurrent=True))
res.append(group("G3 交替异内容(对照)", [P3, Q3, P3, Q3]))
seq_trace("G1' 跨请求逐请求轨迹", P1, n=6)
seq_trace("G3' 异内容交替轨迹（对照组，P 与 Q 轮换）", P3, n=3)
seq_trace("G3' 对照组另一半 Q", Q3, n=3)
print(f"  末态累计: hits={scrape()['hits']:.0f}")
json.dump(res, open(f"/tmp/prefix_disambig_{TAG}.json", "w"), ensure_ascii=False, indent=1)
print(f"  明细已存 /tmp/prefix_disambig_{TAG}.json")
