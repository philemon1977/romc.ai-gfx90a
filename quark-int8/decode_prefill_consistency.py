#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自洽性检验：**decode 一步**算出的分布 是否等于 **把同一个 token 放进 prefill** 算出的分布。

## 为什么这是"判决性"的（不需要任何参考实现）

vLLM 里同一个序列有两条数值路径：
  * prefill：一次前向吃下整段 prompt，第 i 个位置的 logits 由**因果掩码内的全部前文**算出；
  * decode ：每次前向只吃 1 个新 token，全部历史**只能通过 KV cache**（SWA 环 + 压缩 KV +
             indexer 候选/topk 缓冲）进入计算。

把两者对齐就得到一个**纯 API 级**的等价性断言：
    设 greedy 续写为 g1..gK。
    A) prompt = ids            , max_tokens=K → D_A[1..K]（D_A[1]=prefill，D_A[2..K]=decode）
    B) prompt = ids + g1..gi   , max_tokens=1 → D_B[i]（**永远是 prefill**）
    正确实现下必须 D_B[i] ≈ D_A[i+1]。
若 D_A[i+1] 与 D_B[i] 大幅不一致 ⇒ **decode 阶段的 KV/状态通路坏了**：
模型每一步只看得到当前 token（无记忆），于是输出必然锁死成不动点
——这正是"1 1 1 1 1 1…"「我 我 我 我…」这类复读吸引子的签名。

## 额外判据

* `D_A[2]` 与「prompt=[g1] 单 token」的 prefill 分布比较：若二者几乎相同 ⇒ decode 完全无记忆
  （只剩输入 embedding 在起作用，SWA 窗口与压缩 KV 都没读到）。
* `D_A[1]` 对不同 prompt 是否不同：prefill 是否还带提示信息。
* 长 prompt 版本（>1024 token）让 ratio=2 的层**脱离"短上下文快路径"**
  （vLLM: max_seq_len // compress_ratio <= index_topk=512 时用 _fill_short_context_topk_indices
   把 topk 缓冲直接填成 [0..n-1,-1,...]），用于分离"短上下文快路径"的嫌疑。

用法（服务已起在 --port）：
    python3 decode_prefill_consistency.py --port 8119
    python3 decode_prefill_consistency.py --port 8119 --long
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request

MODEL = "/models"
TOPK = 10          # logprobs 取前 10（vLLM 上限 20）
K = 8              # greedy 续写步数


def post(base: str, path: str, body: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize(base: str, text: str, add_special: bool) -> list[int]:
    d = post(base, "/tokenize", {"model": MODEL, "prompt": text,
                                 "add_special_tokens": add_special})
    return list(d["tokens"])


def distribute(base: str, ids: list[int], max_tokens: int) -> list[dict]:
    """返回每个生成位置的 top-k 分布（log-softmax，nats）以及被选中的 token。"""
    d = post(base, "/v1/completions", {
        "model": MODEL, "prompt": ids, "max_tokens": max_tokens,
        "temperature": 0, "logprobs": TOPK,
    })
    ch = d["choices"][0]
    lp = ch.get("logprobs") or {}
    out = []
    for i, tok in enumerate(lp.get("tokens", [])):
        tops = lp["top_logprobs"][i] if i < len(lp.get("top_logprobs", [])) else {}
        out.append({"token": tok, "tokens": list(tops.keys()),
                    "logprobs": {k: float(v) for k, v in tops.items()},
                    "chosen_lp": float(lp["token_logprobs"][i])})
    return out


def cmp_dist(a: dict, b: dict) -> tuple[float, bool, list[str], int]:
    """两个分布的差异。

    ★ 只用**两边都进了 top-k** 的 token 比 —— 早先版本给"缺席 token"填 -30，
      于是"一边有、另一边没有"的普通情形被算成 Δ≈25 nats 的假差异（本次踩过）。
      缺席本身只说明该 token 概率极低，不构成"分布不一致"的证据。
    返回 (共有 token 上的最大 |Δ logprob|, argmax 是否一致, 差异前三, 共有 token 数)。
    """
    shared = set(a["logprobs"]) & set(b["logprobs"])
    diffs = {k: abs(a["logprobs"][k] - b["logprobs"][k]) for k in shared}
    mx = max(diffs.values()) if diffs else float("nan")
    top = sorted(diffs.items(), key=lambda kv: -kv[1])[:3]
    agree = (a["tokens"][0] == b["tokens"][0])
    return mx, agree, [f"{k!r} Δ={v:.2f}" for k, v in top], len(shared)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8119)
    ap.add_argument("--long", action="store_true", help="换成 >1024 token 的长 prompt")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    res: dict = {"port": a.port, "long": a.long, "cases": {}}

    # ---- 提示：① 纯文本 ② 官方 chat 编码（chat 模式，thinking=False）----
    if a.long:
        text = ("The quarterly report covers logistics, staffing, and procurement. " * 60
                + "\n访问码是 ZEPHYR-7781。\n"
                + "Additional operational notes follow. " * 60
                + "\n问题：访问码是什么？只回答访问码。")
    else:
        text = "请用一句话解释什么是量子纠缠，然后给出它的一个日常类比。"

    prompts: dict[str, list[int]] = {}
    prompts["raw"] = tokenize(base, text, True)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "off_enc",
            "/mnt/stripe-3mix-3t2/models/deepseek-ai/DeepSeek-V4.1-Flash/encoding/encoding.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules["off_enc"] = m
        spec.loader.exec_module(m)
        chat = m.encode_messages([{"role": "user", "content": text}],
                                 thinking_mode="chat", drop_thinking=True,
                                 reasoning_effort=None)
        prompts["chat"] = tokenize(base, chat, False)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] chat 提示编码失败（只测 raw）：{e!r}", file=sys.stderr)

    for pname, ids in prompts.items():
        print(f"\n{'='*78}\n[提示 {pname}] prompt_tokens={len(ids)}")
        A = distribute(base, ids, K)
        gtoks = [d["token"] for d in A]
        print(f"[A] prompt=原样, greedy 续写 = {gtoks}")
        print("[A] 逐位 top1 / 其 logprob / top2")
        for i, d in enumerate(A):
            t2 = d["tokens"][1] if len(d["tokens"]) > 1 else None
            print(f"      step{i+1}: {d['token']!r} lp={d['chosen_lp']:.3f} "
                  f"top2={t2!r} lp2={d['logprobs'].get(t2, float('nan')):.3f}")

        # B) 把 greedy 续写"喂回 prefill"逐级复算。
        #    ★ 必须拿到**token id** 才能拼回 prompt：logprobs.tokens 给的是文本，
        #      再去 /tokenize 会引入"文本→id"的不确定性（空格/特殊 token）。
        #      ⇒ 用 vLLM 的 return_token_ids 直接取 id（缺这个字段就放弃 B 臂，不猜）。
        d2 = post(base, "/v1/completions", {"model": MODEL, "prompt": ids,
                                            "max_tokens": K, "temperature": 0,
                                            "logprobs": TOPK, "return_token_ids": True})
        gids = d2["choices"][0].get("token_ids")
        if not gids:
            print("❌ 服务未返回 token_ids（需要 vLLM 支持 return_token_ids）⇒ 无法做 B 臂",
                  file=sys.stderr)
            return 2
        gids = [int(x) for x in gids]
        print(f"[A] greedy token ids = {gids}")
        print("[B] 同一续写用 prefill 复算（每条都是 1 次完整 prefill + 1 步）")
        worst = 0.0
        for i in range(1, len(A)):
            B = distribute(base, ids + list(gids[:i]), 1)
            if not B:
                print(f"      step{i+1}: B 臂无 logprobs，跳过")
                continue
            mx, agree, tops, nsh = cmp_dist(A[i], B[0])
            if mx == mx:  # 非 nan
                worst = max(worst, mx)
            print(f"      step{i+1}: decode[top1={A[i]['token']!r}] vs prefill[top1={B[0]['token']!r}] "
                  f"argmax{'一致' if agree else '★不一致★'} 共有token={nsh} 共有上maxΔlp={mx:.3f}  {tops}")
        print(f"[结论 {pname}] decode-vs-prefill 共有 token 上最大差异 maxΔlp = {worst:.3f} nats "
              f"⇒ {'一致（decode KV 通路正常）' if worst < 0.5 else '★不一致 ⇒ decode 状态通路有问题★'}")

        # C) 无记忆检验：prompt 只剩 g1
        if gids:
            C = distribute(base, [int(gids[0])], 1)
            mx, agree, tops, nsh = cmp_dist(A[1], C[0])
            verdict = ("★decode 几乎无记忆（与'只喂一个 token'等价）★"
                       if (agree and mx < 0.5) else "decode 仍有记忆")
            print(f"[C] prompt=[g1] 单 token 的 prefill 分布 vs A 的 step2："
                  f"argmax{'一致' if agree else '不一致'} 共有token={nsh} 共有上maxΔlp={mx:.3f} {tops}")
            print(f"    ⇒ {verdict}")
        res["cases"][pname] = {"prompt_tokens": len(ids), "greedy": gids,
                               "worst_decode_vs_prefill": worst}
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n已写 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
