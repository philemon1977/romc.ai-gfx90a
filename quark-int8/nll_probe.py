#!/usr/bin/env python3
"""Quality probe for a served model: mean token NLL over a fixed text sample,
via /v1/completions with prompt_logprobs.

⚠️ 只适用于 SPEC<=3（2026-09-17 实撞）：SPEC=5 下本探针会报 ~12.9（≈ln(V)）——
   不是模型坏，是 vLLM 0.28 的 prompt_logprobs 在 MTP 深度 5 下失效（两条路径都失效）。
   探针现在内置闸门：NLL>8 直接拒绝采信并 exit 2。 Used to compare INT8 coverage variants
(experts-only vs experts+attention) on identical input.

Usage: python3 nll_probe.py <port> <model_name> [label]
"""
import json
import math
import os
import sys
import urllib.request

TEXT = (
    "The AMD Instinct MI250X is a dual-die GPU accelerator based on the CDNA 2 architecture, "
    "featuring 128 GB of HBM2e memory and a peak memory bandwidth of roughly 3.2 TB/s per package. "
    "Quantization reduces the number of bits used to represent model weights, which lowers both the "
    "memory footprint and the memory traffic required during autoregressive decoding. For a mixture-of-experts "
    "language model, expert weights dominate the parameter count, so converting only the routed experts to "
    "INT8 typically halves the model size with limited accuracy loss when the weights are quantized per output "
    "channel and activations are quantized dynamically per token. Numerical stability depends on keeping "
    "routers, normalization layers and the small gating projections in higher precision, because their outputs "
    "drive routing decisions and recurrent state updates."
)


def main() -> None:
    port, model = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else model
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps({
            "model": model, "prompt": TEXT, "max_tokens": 1, "temperature": 0,
            "prompt_logprobs": 0,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    pl = d["choices"][0].get("prompt_logprobs")
    if not pl:
        print(f"[{label}] prompt_logprobs unavailable"); return
    lps = []
    for entry in pl[1:]:  # first entry corresponds to the first prompt token
        if entry:
            lps.append(next(iter(entry.values()))["logprob"])
    nll = -sum(lps) / len(lps)
    print(f"[{label}] mean token NLL over {len(lps)} tokens: {nll:.4f}  (perplexity {math.exp(nll):.2f})")

    # ── 自检闸门（2026-09-17 实撞后加）──────────────────────────────────────────
    # 本机 vLLM(0.28) 在 **SPEC=5（MTP 深度 5）** 下 `prompt_logprobs` 与 `echo+logprobs`
    # 两条路径**都**给出近似均匀随机的 logprob ⇒ NLL ≈ ln(vocab) ≈ 12.9。
    # 实测边界：SPEC=0 → 1.9638 ✅、SPEC=3 → 1.9654 ✅、SPEC=5 → 12.92 ❌；
    # 与前缀缓存无关（开关都一样）、抬高 --max-num-batched-tokens 也无效。
    # 因此：NLL 明显超过"均匀分布"量级时，拒绝采信，避免污染质量结论。
    UNIFORM = math.log(150000)          # ≈ 11.9，词表量级
    if nll > 8.0:
        print(f"[{label}] ⛔ 拒绝采信：NLL={nll:.4f} 已接近均匀分布量级（ln(V)≈{UNIFORM:.1f}）"
              f"，几乎必然是 probe/配方不兼容（已知：SPEC=5 触发）。")
        print(f"[{label}]    处置：改在 SPEC=0 或 SPEC=3 下测质量（两者均已复核为 ~1.96），"
              f"速度仍可在 SPEC=5 下测。")
        sys.exit(2)

    out = {"label": label, "tokens": len(lps), "mean_nll": nll, "ppl": math.exp(nll)}
    # 落盘到脚本同目录（原先写死 /work/ = 容器时代路径，裸机下必 FileNotFoundError；
    # 2026-09-17 实撞：NLL 已算出并打印，却在写盘时崩掉 ⇒ 读数在 stdout，别只看文件）
    outdir = os.environ.get("NLL_OUTDIR") or os.path.dirname(os.path.abspath(__file__))
    outpath = os.path.join(outdir, f"NLL_{label}.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{label}] 已写 {outpath}")


if __name__ == "__main__":
    main()
