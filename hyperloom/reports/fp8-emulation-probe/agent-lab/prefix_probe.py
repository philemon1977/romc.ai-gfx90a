#!/usr/bin/env python3
"""Agentic (vibe-coding) turn probe: long shared prefix, one session at a time.

The real shape of an agent loop is not "64 independent requests"; it is a
sequence of turns that re-send a mostly-unchanged context (system prompt + repo
+ prior turns) and append a small delta. So the metrics that matter are:

  * TTFT of turn 1 (cold prefix, full prefill of ``--context`` tokens)
  * TTFT of turns 2..N (prefix-cache hit on the shared part)
  * decode tok/s at batch 1 (turn speed, unaffected by cache)
  * e2e latency per turn

Run inside hyperloom-srv:

    /opt/envs/vllm/bin/python3 prefix_probe.py --base-url http://127.0.0.1:8899 \
        --model <path> --out prefix_probe.json --context 32768 --turns 4 --osl 256
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

FILLER = (
    "def reconcile_ledger(entries, *, tol=1e-6): "
    "    total = sum(e.amount for e in entries if abs(e.amount) > tol) "
    "    return round(total, 6)  # keep the drift below float noise "
)
DELTA = (
    "Now review this file and list the three highest-risk defects with a one-line fix for each. "
    f"Turn marker: "
)


def build_context(target_chars: int) -> str:
    reps = max(1, target_chars // len(FILLER) + 1)
    return (FILLER * reps)[:target_chars]


async def one_turn(
    client: httpx.AsyncClient, base_url: str, model: str, prompt: str, osl: int, tag: str
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": osl,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    t0 = time.perf_counter()
    ttft = first = last = None
    n_tok = 0
    usage: dict = {}
    async with client.stream("POST", f"{base_url}/v1/completions", json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if not body or body == "[DONE]":
                continue
            chunk = json.loads(body)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                if not (choice.get("text") or ""):
                    continue
                now = time.perf_counter()
                if ttft is None:
                    ttft, first = now - t0, now
                last = now
                n_tok += 1
    end = time.perf_counter()
    decode_s = (last - first) if (first and last and last > first) else None
    out = {
        "tag": tag,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens") or n_tok,
        "ttft_ms": round(ttft * 1000, 1) if ttft is not None else None,
        "decode_tps": round((n_tok - 1) / decode_s, 2) if decode_s and n_tok > 1 else None,
        "tpot_ms": round(decode_s * 1000 / max(1, n_tok - 1), 1) if decode_s else None,
        "e2e_s": round(end - t0, 2),
    }
    print(json.dumps(out), flush=True)
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=int, default=32768, help="approx shared-prefix tokens")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--osl", type=int, default=256)
    args = ap.parse_args()

    # ~4 chars/token on code-ish text; the exact length is read back from usage.
    base = build_context(args.context * 4)
    turns = []
    history = ""
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        for i in range(args.turns):
            prompt = base + history + DELTA + f"{i:03d}"
            turns.append(await one_turn(client, args.base_url, args.model, prompt, args.osl, f"turn{i+1}"))
            history += f"\nassistant: (turn {i+1} answer placeholder) " + ("x" * 200)

    cold = turns[0]["ttft_ms"]
    warm = [t["ttft_ms"] for t in turns[1:] if t["ttft_ms"]]
    decodes = [t["decode_tps"] for t in turns if t["decode_tps"]]
    summary = {
        "context_tokens_measured": turns[0]["prompt_tokens"],
        "cold_ttft_ms": cold,
        "warm_ttft_ms_median": statistics.median(warm) if warm else None,
        "prefix_cache_speedup": round(cold / statistics.median(warm), 2) if (cold and warm) else None,
        "decode_tps_median": statistics.median(decodes) if decodes else None,
        "turns": turns,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "turns"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
