#!/usr/bin/env python3
"""8107 的"问题2"验证探针：四种模式，全部用定稿口径。

MODE=consistency  —— **数值硬门**：同一长 prompt 连发 3 次（冷/暖/暖），
                     token 序列必须逐位相同、每 token logprob 最大差 ≤ 容差。
                     这是"递归状态复用静默算错"的唯一检测手段。
                     ⚠️ 用**输出 logprobs**（不用 prompt_logprobs：本会话在 SPEC=5 上实撞过污染）。
MODE=multiturn    —— agent 式多轮（上下文逐轮累积）：每轮报 prompt tok / 命中 tok / 期望命中 / TTFT / decode。
                     期望命中上限 = (floor(共享长度/block) - 1) * block（block=400，日志实测平台对齐值）。
MODE=singlestream —— 单流 count prompt n=5（回归检查：改前 92.07 t/s）。
用法：qwen38_agent_probe.py PORT MODEL MODE [TURNS] [ANS_TOK]
"""
import json
import re
import statistics
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8107"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-flash-next"
MODE = sys.argv[3] if len(sys.argv) > 3 else "consistency"
TURNS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
ANS = int(sys.argv[5]) if len(sys.argv) > 5 else 400
BLOCK = 400
TOL = 0.2   # logprob 容差：**必须按引擎先天抖动定**。实测本引擎(8107/bf16/MoE)在无任何复用、同 prompt 重复请求下，token 逐位相同但 |Δlogprob| 可达 0.11 ⇒ 用 0.01 会给出假 FAIL。判据主项是「token 逐位相同」。

CNT = "Count slowly from one to ninety, writing each number in words on its own line."
BODY = ("The following is a repository snapshot. Read it carefully before answering. "
        "launcher.sh sets ROCM_PATH, MODEL_PATH and GPU_MEM_UTIL, then execs vllm. "
        "kernel.py registers a gate GEMV kernel under flag MI250_GATE_GEMV. "
        "ledger.jsonl records every experiment verdict with a doc pointer. ")
TAIL = ("\nQuestion: in one short sentence, what does MI250_GATE_GEMV change, and why must the "
        "other kernel flags stay off? Answer now.")
SYS_AGENT = ("You are a coding agent working inside a repository. Tools: read_file, write_file, "
             "run_shell, grep. Inspect before editing; keep changes minimal; after editing run the "
             "tests and report exactly what changed. ") * 12
ASK = "\nUser: explain what the file mi250_kernels.py does, step by step, then answer follow-ups.\n"
FU = "\nUser: now give the exact shell command to enable only the gate GEMV flag, and why the others stay off.\n"


def scrape():
    t = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=60).read().decode()
    o = {}
    for k in ("prefix_cache_queries", "prefix_cache_hits"):
        m = re.search(rf"^vllm:{k}_total\{{[^}}]*\}}\s+([0-9.eE+]+)$", t, re.M)
        o[k] = float(m.group(1)) if m else 0.0
    return o


def gen(prompt, maxtok, logprobs=False):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": maxtok, "temperature": 0,
            "ignore_eos": True, "stream": True}
    if logprobs:
        body["logprobs"] = 1
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    toks, lps, text = [], [], []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data: ") or b"[DONE]" in line:
                continue
            if ttft is None:
                ttft = time.time() - t0
            try:
                d = json.loads(line[6:])
            except Exception:
                continue
            ch = d.get("choices", [{}])[0]
            if ch.get("text"):
                text.append(ch["text"])
            lp = ch.get("logprobs") or {}
            for t_, l_ in zip(lp.get("tokens") or [], lp.get("token_logprobs") or []):
                toks.append(t_)
                lps.append(l_)
    return ttft, time.time() - t0, "".join(text), toks, lps


def mode_consistency():
    p = BODY * 55 + TAIL
    print(f"[consistency] prompt ≈ {len(p)//4} tok，连发 3 次（冷/暖/暖），容差 {TOL}")
    runs = []
    for i in range(1, 4):
        c0 = scrape()
        ttft, dt, text, toks, lps = gen(p, 64, logprobs=True)
        c1 = scrape()
        runs.append((toks, lps))
        print(f"  run{i}: hits +{int(c1['prefix_cache_hits']-c0['prefix_cache_hits'])}/"
              f"{int(c1['prefix_cache_queries']-c0['prefix_cache_queries'])}  "
              f"TTFT {ttft:.2f}s  {len(lps)} 个 logprob", flush=True)
    base_t, base_l = runs[0]
    ok, worst = True, 0.0
    for i, (t, l) in enumerate(runs[1:], start=2):
        same = t == base_t
        ok &= same
        d = max([abs(a - b) for a, b in zip(base_l, l) if a is not None and b is not None] or [0.0])
        worst = max(worst, d)
        print(f"  run1 vs run{i}: token {'相同 ✅' if same else '不同 ❌'}  最大|Δlogprob| {d:.3e}")
    verdict = "PASS ✅" if (ok and worst <= TOL) else "FAIL ❌"
    print(f"[consistency] 数值硬门：{verdict}（token 逐位相同 且 |Δlogprob| ≤ {TOL}；实测 {worst:.3e}）")
    return 0 if verdict.startswith("PASS") else 2


def mode_multiturn():
    conv = SYS_AGENT + ASK
    print(f"[multiturn] {TURNS} 轮 × {ANS} tok；block={BLOCK}；期望命中上限=(floor(共享/{BLOCK})−1)×{BLOCK}")
    print(f"  {'轮':>3} {'prompt':>8} {'命中':>7} {'共享≈':>7} {'期望':>7} {'TTFT s':>7} {'decode t/s':>11}")
    prev = 0
    for turn in range(1, TURNS + 1):
        c0 = scrape()
        ttft, dt, ans, _, _ = gen(conv, ANS)
        c1 = scrape()
        dq = c1["prefix_cache_queries"] - c0["prefix_cache_queries"]
        dh = c1["prefix_cache_hits"] - c0["prefix_cache_hits"]
        shared = prev
        exp = max(0, (int(shared // BLOCK) - 1) * BLOCK) if shared else 0
        dec = ANS / max(dt - ttft, 1e-3)
        print(f"  {turn:>3} {int(dq):>8} {int(dh):>7} {int(shared):>7} {exp:>7} {ttft:>7.2f} {dec:>11.2f}",
              flush=True)
        conv += ans + FU
        prev = int(dq) + len(ans) // 4
    return 0


def mode_singlestream():
    gen(CNT, 256)
    ts = []
    for _ in range(5):
        _, dt, _, _, _ = gen(CNT, 256)
        ts.append(256 / dt)
    med = statistics.median(ts)
    print(f"[singlestream] MEDIAN {med:.2f} t/s  min {min(ts):.2f} max {max(ts):.2f} "
          f"spread {100*(max(ts)-min(ts))/med:.1f}%（n=5，丢首个；改前基线 92.07）")
    return 0


def mode_consistency2():
    """修好设计的数值硬门：先打一个无关请求预热（**排除"首个请求"口径**），
    再用**唯一 salt 的 prompt** 做 冷→暖→暖 三次比对。
    为什么必须这样做：本会话实测第一个请求与后续请求不在同一口径（kernel/图选择不同），
    拿它当基准会得到假 FAIL——这正是上一版探针的错。"""
    gen("Say OK.", 4)                      # 预热（丢弃）
    q = BODY * 55 + TAIL + f"\n[salt {time.time_ns()}]"
    print(f"[consistency2] 已预热；prompt ≈ {len(q)//4} tok；冷→暖→暖")
    runs = []
    for i in range(1, 4):
        c0 = scrape()
        ttft, dt, text, toks, lps = gen(q, 64, logprobs=True)
        c1 = scrape()
        runs.append((toks, lps))
        print(f"  run{i}: hits +{int(c1['prefix_cache_hits']-c0['prefix_cache_hits'])}/"
              f"{int(c1['prefix_cache_queries']-c0['prefix_cache_queries'])}  TTFT {ttft:.2f}s", flush=True)
    base_t, base_l = runs[0]
    ok, worst = True, 0.0
    for i, (t, l) in enumerate(runs[1:], start=2):
        same = t == base_t
        d = max([abs(a - b) for a, b in zip(base_l, l) if a is not None and b is not None] or [0.0])
        worst = max(worst, d)
        print(f"  run1(冷) vs run{i}(暖): token {'相同 ✅' if same else '不同 ❌'}  最大|Δlogprob| {d:.3e}")
        ok &= same
    # 暖 vs 暖 也要自洽（否则说明引擎本身不确定，不能把差异归给"状态复用"）
    same_w = runs[1][0] == runs[2][0]
    print(f"  暖 vs 暖: token {'相同 ✅' if same_w else '不同 ❌'}")
    verdict = "PASS ✅" if (ok and worst <= TOL) else "FAIL ❌"
    print(f"[consistency2] 数值硬门：{verdict}（判据：冷 vs 暖 token 逐位相同 且 |Δlogprob| ≤ {TOL}；实测 {worst:.3e}）")
    return 0 if verdict.startswith("PASS") else 2


def mode_consistency3():
    """真正的 冷→暖→暖：salt 放在**最前面** ⇒ 整段前缀都是新的 ⇒ run1 是货真价实的冷算。
    （consistency2 的 salt 在尾部，前缀仍命中 ⇒ run1 其实也是暖的，没有冷基准。）"""
    gen("Say OK.", 4)
    q = f"[salt {time.time_ns()}] " + BODY * 55 + TAIL
    print(f"[consistency3] 已预热；前缀唯一（salt 在首）；prompt ≈ {len(q)//4} tok")
    runs = []
    for i in range(1, 4):
        c0 = scrape()
        ttft, dt, text, toks, lps = gen(q, 64, logprobs=True)
        c1 = scrape()
        runs.append((toks, lps))
        print(f"  run{i}: hits +{int(c1['prefix_cache_hits']-c0['prefix_cache_hits'])}/"
              f"{int(c1['prefix_cache_queries']-c0['prefix_cache_queries'])}  TTFT {ttft:.2f}s", flush=True)
    (bt, bl), (wt, wl) = runs[0], runs[1]
    d_cw = max([abs(a - b) for a, b in zip(bl, wl) if a is not None and b is not None] or [0.0])
    same_cw = bt == wt
    same_ww = runs[1][0] == runs[2][0]
    print(f"  冷 vs 暖: token {'相同 ✅' if same_cw else '不同 ❌'}  最大|Δlogprob| {d_cw:.3e}")
    print(f"  暖 vs 暖: token {'相同 ✅' if same_ww else '不同 ❌'}")
    verdict = "PASS ✅" if (same_cw and same_ww and d_cw <= TOL) else "FAIL ❌"
    print(f"[consistency3] 数值硬门：{verdict}（冷vs暖 token 逐位相同 且 |Δlogprob| ≤ {TOL}；实测 {d_cw:.3e}）")
    return 0 if verdict.startswith("PASS") else 2


if __name__ == "__main__":
    sys.exit({"consistency": mode_consistency, "consistency2": mode_consistency2, "consistency3": mode_consistency3, "multiturn": mode_multiturn,
              "singlestream": mode_singlestream}[MODE]())
