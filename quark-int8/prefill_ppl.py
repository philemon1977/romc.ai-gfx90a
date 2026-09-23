#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prefill-only 尺子：把一段**自然文本**当 prompt，用 API 的 prompt_logprobs 读模型对每个位置的
   下一 token 分布，得到

     * PPL      = exp(-mean(log p(实际下一个 token)))   —— 健康的 552B 模型在自然文本上应是个位数
     * top1 命中率 = 模型 argmax 恰等于实际下一个 token 的比例
     * 逐位置曲线 —— 前 N 个位置打印，看是"from the start 就平"还是"中途塌"

为什么要它：`eval_quality_ab.py` 量的是**生成**结果（decode 主导），而 layer0/2 的对拍量的是
**单层前向**。中间缺一个"整段前向是否健康"的整体量尺。PPL 高到离谱（例如 1e4）
⇒ 整段 prefill 就是坏的，不必再去怀疑 decode；PPL 正常 ⇒ 病灶在 decode/状态通路。

用法：
    python3 prefill_ppl.py --port 8119
    python3 prefill_ppl.py --port 8119 --text-file /path/to.txt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import urllib.request

MODEL = "/models"

# 默认语料：模型的 README（英文，真实自然语言）+ 本仓的中文文档片段（保证中英都覆盖）
DEFAULT_SOURCES = [
    ("/mnt/stripe-3mix-3t2/models/deepseek-ai/DeepSeek-V4.1-Flash/README.md", 0,
     "英文·模型 README"),
    ("/mnt/stripe-3mix-3t2/models/deepseek-ai/DeepSeek-V4.1-Flash/README.md", 6000,
     "英文·README 另一段"),
]


def post(base: str, path: str, body: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize(base: str, text: str) -> list[int]:
    d = post(base, "/tokenize", {"model": MODEL, "prompt": text})
    return list(d["tokens"])


def measure(base: str, ids: list[int], tag: str) -> dict:
    # max_tokens=1：只采样 1 个 token（vLLM 不接受 0），prompt_logprobs 仍覆盖整段 prompt
    body = {"model": MODEL, "prompt": ids, "max_tokens": 1,
            "prompt_logprobs": 5, "echo": False}
    d = post(base, "/v1/completions", body)
    plp = d["choices"][0].get("prompt_logprobs")
    if plp is None:
        return {"tag": tag, "error": "服务未返回 prompt_logprobs"}
    lps, hits, detail = [], 0, []
    pos_lp: list[tuple[int, float, int]] = []   # (位置, logprob, rank)
    for i, ent in enumerate(plp):
        if i == 0 or not ent:
            continue                      # 位置 0 没有前文
        # ent: {token_id(str/int): {"logprob":..,"rank":..}} 或 None
        items = []
        for k, v in ent.items():
            if isinstance(v, dict) and "logprob" in v:
                items.append((int(k), float(v["logprob"]), v.get("rank")))
        if not items:
            continue
        # 实际 token = 下一个位置的 id
        nxt = int(ids[i])
        got = [(t, lp, r) for (t, lp, r) in items if t == nxt]
        if not got:
            continue
        lp = got[0][1]
        lps.append(lp)
        top1 = min(items, key=lambda x: -(x[1]))[0]
        rank = got[0][2]
        pos_lp.append((i, lp, int(rank) if rank is not None else -1))
        if rank == 1 or top1 == nxt:
            hits += 1
        if len(detail) < 24:
            detail.append((i, nxt, round(lp, 3), rank, top1))
    if not lps:
        return {"tag": tag, "error": "没有可用的位置对数"}
    mean_lp = sum(lps) / len(lps)
    ppl = math.exp(-mean_lp)
    r = {"tag": tag, "n_pos": len(lps), "mean_logprob": round(mean_lp, 4),
         "ppl": round(ppl, 2), "top1_hit_rate": round(hits / len(lps), 4),
         "first24": detail}
    print(f"[{tag}] 位置数={r['n_pos']} 平均 logprob={r['mean_logprob']} "
          f"⇒ PPL={r['ppl']}  top1 命中率={r['top1_hit_rate']}")
    print(f"      前 {len(detail)} 个位置 (位置, 实际token, 其logprob, 其排名, 模型top1):")
    for row in detail:
        print(f"        pos={row[0]:4d} tok={row[1]:6d} lp={row[2]:8.3f} rank={row[3]} top1={row[4]}")
    # 分桶：每 100 个位置一段。用来判"是不是跨过某个长度阈值之后才塌"
    # （vLLM 的 indexer 在 max_seq_len//compress_ratio <= index_topk=512 时走"短上下文快路径"，
    #   ratio=2 的层在 1024 token 处切换，ratio=1 的层在 512 token 处切换）。
    buckets = []
    for lo in range(0, len(pos_lp), 100):
        seg = [x for x in pos_lp if lo <= x[0] < lo + 100]
        if not seg:
            continue
        mlp = sum(x[1] for x in seg) / len(seg)
        hr = sum(1 for x in seg if x[2] == 1) / len(seg)
        buckets.append({"lo": lo, "n": len(seg), "mean_lp": round(mlp, 3),
                        "ppl": round(math.exp(-mlp), 2), "top1": round(hr, 3),
                        "median_rank": sorted(x[2] for x in seg)[len(seg) // 2]})
    print("      分桶（每 100 位置）: 段起点 mean_logprob PPL top1命中 排名中位")
    for b in buckets:
        print(f"        pos {b['lo']:5d}-{b['lo']+b['n']-1:<5d} lp={b['mean_lp']:7.3f} "
              f"PPL={b['ppl']:8.2f} top1={b['top1']:.3f} 排名中位={b['median_rank']}")
    r["buckets"] = buckets
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8119)
    ap.add_argument("--text-file", default="")
    ap.add_argument("--max-tokens", type=int, default=600, help="每个语料取多少 token")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    res = {"port": a.port, "items": []}

    srcs = ([(a.text_file, 0, os.path.basename(a.text_file))]
            if a.text_file else DEFAULT_SOURCES)
    for path, off, tag in srcs:
        try:
            raw = open(path, encoding="utf-8", errors="replace").read()
        except OSError as e:
            print(f"[{tag}] 读不到 {path}: {e}", flush=True)
            continue
        seg = raw[off:off + 6000]
        ids = tokenize(base, seg)[: a.max_tokens]
        print(f"\n=== [{tag}] {path} off={off} → {len(ids)} token", flush=True)
        print("    前 120 字符:", repr(seg[:120]))
        r = measure(base, ids, tag)
        res["items"].append(r)
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n已写 {a.out}")
    print("\n[判读] PPL 个位数~十几 = prefill 健康；PPL 成百上千 = 整段前向就是坏的。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
