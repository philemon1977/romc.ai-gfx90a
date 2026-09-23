#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""针尖检索的**分布级**量尺：不问模型"生成"了什么，只问
"在'访问码是 ___'这个位置上，模型给正确答案的概率排第几"。

为什么要它：生成结果会被 decode/采样/复读掩盖；而 prompt_logprobs 直接给出
**prefill 时该位置的分布**。配合一个对照臂（把上下文里的答案换成另一个串），
就能把"真的从上下文里取到了"与"瞎猜/复读"分开：

   A 臂：上下文含真实访问码 ZEPHYR-7781
   B 臂：同样文本，但把出现过的访问码换成 ZEPHYR-9999（并让问题里的期望答案不变）
   ⇒ 若 A 臂里 ZEPHYR-7781 的 logprob ≫ B 臂，说明模型确实在读上下文（检索在）；
      若两臂几乎相同且都很低，说明模型根本没把上下文用于这个位置。

用法： python3 needle_logprob.py --port 8119
"""
from __future__ import annotations

import argparse
import json
import urllib.request

MODEL = "/models"


def post(base: str, path: str, body: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize(base: str, text: str) -> list[int]:
    return list(post(base, "/tokenize", {"model": MODEL, "prompt": text})["tokens"])


def _find_last_subseq(ids: list[int], sub: list[int]) -> int:
    """返回 sub 在 ids 中**最后一次**出现的起始下标；找不到返回 -1。"""
    for i in range(len(ids) - len(sub), -1, -1):
        if ids[i:i + len(sub)] == sub:
            return i
    return -1


def probe(base: str, text: str, watch: str, tag: str) -> dict:
    """返回 watch 这个字符串的各 token 在各自位置上的 logprob / rank。

    ★ 不能用"字符偏移 → tokenize 前缀取长度"来定位：BPE 会把前导空格并进相邻 token，
      字符边界与 token 边界不对齐（本脚本第一版就因此错位，只命中 1 个无关 token）。
      改为把 watch 的 token 序列在整段 ids 里做**子序列查找**（含"带前导空格"变体）。
    """
    ids = tokenize(base, text)
    start, w_ids = -1, None
    for cand in (tokenize(base, " " + watch), tokenize(base, watch)):
        i = _find_last_subseq(ids, cand)
        if i >= 0:
            start, w_ids = i, cand
            break
    if start < 0:
        raise RuntimeError(f"在 ids 里找不到 {watch!r} 的 token 子序列")
    plp = post(base, "/v1/completions", {"model": MODEL, "prompt": ids,
                                        "max_tokens": 1, "prompt_logprobs": 20})["choices"][0]["prompt_logprobs"]
    rows = []
    for j, tid in enumerate(w_ids):
        idx = start + j
        ent = plp[idx] if idx < len(plp) else None
        if not ent:
            rows.append({"tok": tid, "logprob": None, "rank": None})
            continue
        v = ent.get(str(tid)) or ent.get(tid)
        rows.append({"tok": tid,
                     "logprob": float(v["logprob"]) if v else None,
                     "rank": int(v["rank"]) if v and v.get("rank") is not None else None,
                     "decoded": (v or {}).get("decoded_token")})
    avg = [r["logprob"] for r in rows if r["logprob"] is not None]
    res = {"tag": tag, "prompt_tokens": len(ids), "watch": watch,
           "watch_token_ids": w_ids, "start_idx": start, "rows": rows,
           "mean_logprob": round(sum(avg) / len(avg), 3) if avg else None}
    print(f"[{tag}] prompt={len(ids)} tok，被观察串 {watch!r} 的 token={w_ids}")
    for r in rows:
        print(f"        tok={r['tok']:6d} logprob={r['logprob']} rank={r['rank']} {r.get('decoded')!r}")
    print(f"        ⇒ 平均 logprob = {res['mean_logprob']}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8119)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"

    filler = ("The quarterly report covers logistics, staffing, and procurement. " * 20)
    code = "ZEPHYR-7781"
    other = "ZEPHYR-9999"
    # A 臂：上下文里出现过真码，问句要求答它
    A = (f"阅读下面的材料，然后回答问题。\n\n{filler}\n访问码是 {code}。\n\n"
         f"问题：访问码是什么？\n答：访问码是 ")
    # B 臂：上下文里出现的是**另一个**码，但结尾同样写出真码。
    #   ★ 只有两臂的**测量点文本完全相同**（都以真码结尾），才能在同一位形上比较 logprob。
    B = (f"阅读下面的材料，然后回答问题。\n\n{filler}\n访问码是 {other}。\n\n"
         f"问题：访问码是什么？\n答：访问码是 {code}。")
    A = A + code + "。"          # A 臂结尾同样写出真码（测量点一致）
    res = {"port": a.port, "items": []}
    res["items"].append(probe(base, A, code, "A 上下文含真码 → 看真码"))
    res["items"].append(probe(base, B, code, "B 上下文含他码 → 看真码(对照)"))
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n已写 {a.out}")
    print("\n[判读] A 臂真码的 logprob 明显高于 B 臂 → 检索在；两臂都低且相近 → 上下文没被用上。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
