#!/usr/bin/env python3
"""Task ①: measure real MoE routing hotness on the served Ornith-1.5-397B INT8 model.

The server must be started with `--enable-return-routed-experts`; each completion
response then carries `choices[0]["routed_experts"]` of shape
[seq_len, num_layers, topk].

Outputs:
  routing_samples.npz  raw routing tensors (per request)
  ROUTING_REPORT.md    hotness distribution, pool-capacity -> hit-rate curve,
                       consecutive-token locality, and the DRAM/PCIe cost model

Key constants for this model (see RESULT.md):
  experts/layer = 512, layers = 60, topk = 10
  expert size  = 12.58 MB int8 (full)  -> 1.572 MB per rank at TP/EP8
  PCIe measured = 26.7 GB/s H2D (pinned)
  decode budget measured = 26.6 ms/token  -> 0.44 ms per layer
"""
import base64
import io
import json
import os
import sys
import time
import urllib.request

import numpy as np

PORT = int(os.environ.get("PORT", "8100"))
MODEL = os.environ.get("MODEL", "Ornith-probe")
N_REQ = int(os.environ.get("N_REQ", "24"))
MAX_TOK = int(os.environ.get("MAX_TOK", "128"))
OUT_DIR = os.environ.get("OUT_DIR", "/work")

PROMPTS = [
    "Explain how tensor parallelism shards a mixture-of-experts layer and what communication it requires.",
    "Write a short technical summary of HBM2e versus HBM3 memory bandwidth and capacity tradeoffs.",
    "请用中文解释为什么 MoE 模型中专家权重的内存带宽是解码阶段的主要瓶颈。",
    "Describe the difference between INT8 per-channel weight quantization and per-tensor quantization.",
    "Summarize the architecture of the AMD CDNA2 GPU and how it differs from CDNA3.",
    "Write Python code that computes a running mean and variance in a streaming fashion.",
    "Explain speculative decoding and why its acceptance rate matters for throughput.",
    "分析一下长上下文推理中 KV cache 的显存占用如何随序列长度增长。",
    "Describe how paged attention manages KV cache memory allocation.",
    "What are the tradeoffs of expert parallelism versus tensor parallelism for large MoE models?",
    "Explain the role of the router in a mixture-of-experts transformer layer.",
    "Write a paragraph about the physics of photolithography at the 3nm node.",
    "解释为什么量化后再做校准会带来精度损失，以及 per-token 动态量化如何避免它。",
    "Summarize the history of the RISC-V instruction set architecture.",
    "Explain how CUDA graphs reduce kernel launch overhead during LLM decoding.",
    "Describe the mathematical definition of the gated delta rule used in linear attention.",
]


def request(prompt: str) -> dict:
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOK,
        "temperature": 0,
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.load(r)


def main() -> None:
    samples = []
    t0 = time.time()
    for i in range(N_REQ):
        prompt = PROMPTS[i % len(PROMPTS)]
        d = request(prompt)
        re_arr = d["choices"][0].get("routed_experts")
        if re_arr is None:
            print("routed_experts missing -> is --enable-return-routed-experts on?")
            sys.exit(2)
        # the API returns base64-encoded .npy bytes (uint16, [seq_len, layers, topk])
        a = np.load(io.BytesIO(base64.b64decode(re_arr))) if isinstance(re_arr, str) else np.asarray(re_arr)
        a = a.astype(np.int32)
        samples.append(a)
        tok = d["usage"]["completion_tokens"]
        print(f"[{i+1}/{N_REQ}] got routing {a.shape} (tokens={tok}) in {time.time()-t0:.0f}s total")
    np.savez_compressed(os.path.join(OUT_DIR, "routing_samples.npz"),
                        **{f"s{i}": a for i, a in enumerate(samples)})

    allr = np.concatenate([a.reshape(-1, a.shape[-2], a.shape[-1]) for a in samples], axis=0)
    T, L, K = allr.shape
    print(f"collected {T} token-positions x {L} layers x top{K}")

    # per-layer frequency distribution
    per_layer_counts = np.zeros((L, 512), dtype=np.int64)
    for l in range(L):
        ids, cnt = np.unique(allr[:, l, :], return_counts=True)
        per_layer_counts[l, ids] = cnt
    tot_per_layer = per_layer_counts.sum(axis=1)

    caps = [8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 448]
    rows = []
    for C in caps:
        hit = []
        for l in range(L):
            order = np.argsort(-per_layer_counts[l])
            hit.append(per_layer_counts[l][order[:C]].sum() / tot_per_layer[l])
        hit = float(np.mean(hit))
        off_bytes_card = (512 - C) * 1.572e6 * L / 1e9          # GB freed per card (TP8 layout)
        miss_inst = (1 - hit) * K                                # experts/token/layer fetched from DRAM
        dram_tok = miss_inst * 1.572e6 * L / 1e9                 # GB per token per rank
        pcie_ms = dram_tok / 26.7 * 1000
        per_layer_ms = pcie_ms / L
        rows.append((C, hit, off_bytes_card, dram_tok * 1000, pcie_ms, per_layer_ms))

    # consecutive-token locality (prefetch predictability)
    loc = []
    for a in samples:
        if a.shape[0] < 2:
            continue
        for l in range(L):
            prev = [set(x) for x in a[:-1, l, :]]
            cur = [set(x) for x in a[1:, l, :]]
            loc.append(np.mean([len(p & c) / K for p, c in zip(prev, cur)]))
    locality = float(np.mean(loc)) if loc else float("nan")

    # global (cross-layer) skew as well
    flat = allr.reshape(-1)
    _, cnts = np.unique(flat, return_counts=True)
    cnts = np.sort(cnts)[::-1]
    cum = np.cumsum(cnts) / cnts.sum()

    with open(os.path.join(OUT_DIR, "ROUTING_REPORT.md"), "w") as f:
        f.write("# Task ① MoE 路由热度实测（Ornith-1.5-397B INT8）\n\n")
        f.write(f"样本：{T} tokens × {L} 层 × top{K} → {T*L*K:,} 个专家实例；"
                f"采集耗时 {time.time()-t0:.0f}s\n\n")
        f.write(f"**相邻 token 路由重合度（top-10 交集/10）= {locality:.3f}**"
                f" → 用上一 token 路由做预取预测的期望命中率量级\n\n")
        f.write("## 每层专家池容量 → 命中率 → 代价\n\n")
        f.write("| 池容量(每层常驻专家) | 卸载比例 | 命中率 | 释放显存/卡 | DRAM流量/token/卡 | PCIe时间/token | 折合每层 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for C, hit, freed, dram_mb, pcie_ms, per_layer in rows:
            f.write(f"| {C} | {(512-C)/512*100:.0f}% | {hit*100:.1f}% | {freed:.1f} GB | "
                    f"{dram_mb:.0f} MB | {pcie_ms:.1f} ms | {per_layer*1000:.0f} µs |\n")
        f.write("\n（层预算参考：当前 26.6 ms/token ÷ 60 层 = **0.44 ms/层**）\n\n")
        f.write("## 全局（跨层共享专家 ID 空间）集中度\n\n")
        for frac in (0.05, 0.1, 0.2, 0.3, 0.5):
            idx = int(len(cum) * frac) - 1
            f.write(f"- 最热的 {frac*100:.0f}% 实例所覆盖的专家数占比：{cum[max(idx,0)]*100:.1f}% (累计计数的位置)\n")
        f.write("\n## 结论要点\n\n")
        f.write(f"- 相邻 token 重合度 {locality:.3f}：预取预测可覆盖约 {locality*100:.0f}% 的需求，"
                f"剩余 {100-locality*100:.0f}% 需靠 cache 命中或实时抓取。\n")
        f.write("- 命中率曲线用于决定「卸载多少冷专家能换到多少 GB」以及对应的 PCIe 代价。\n")
    print(open(os.path.join(OUT_DIR, "ROUTING_REPORT.md")).read())
    print(json.dumps({"report": os.path.join(OUT_DIR, "ROUTING_REPORT.md"),
                      "samples": os.path.join(OUT_DIR, "routing_samples.npz")}))


if __name__ == "__main__":
    main()
