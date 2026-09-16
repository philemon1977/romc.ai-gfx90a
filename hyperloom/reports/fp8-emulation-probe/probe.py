#!/usr/bin/env python3
"""Single-stream + low-concurrency probe for the gfx90a FP8->BF16 emulation route.

Runs INSIDE hyperloom-srv (that is where /opt/envs/vllm + the ROCm stack live):

    /opt/envs/vllm/bin/python probe.py --base-url http://127.0.0.1:8891 --out results.json

Measures, at ISL ~1024:
  * single stream x3, OSL 256  -> TTFT, decode tok/s, TPOT, e2e
  * concurrency 8  and 32, OSL 128 -> aggregate output tok/s, mean/p50 TTFT, mean TPOT

Writes JSON so the numbers can be diffed against `llama.cpp Q8_0 = 47.28 tok/s`
(the in-service route on the same 8x MI250X) without re-reading logs.
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
    "The quick brown fox jumps over the lazy dog while the observatory tracks "
    "a satellite pass and the operator annotates each frame with its timestamp. "
)


def build_prompt(target_chars: int = 4096) -> str:
    return (FILLER * (target_chars // len(FILLER) + 1))[:target_chars]


async def one_stream(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    tag: str,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    t0 = time.perf_counter()
    ttft = None
    t_first = t_last = None
    n_tok = 0
    usage = {}
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
                text = choice.get("text") or ""
                if not text:
                    continue
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                    t_first = now
                t_last = now
                n_tok += 1
    t_end = time.perf_counter()
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens") or n_tok
    decode_tps = None
    tpot_ms = None
    if t_first is not None and t_last is not None and t_last > t_first and completion_tokens > 1:
        span = t_last - t_first
        decode_tps = (completion_tokens - 1) / span
        tpot_ms = span / (completion_tokens - 1) * 1000.0
    return {
        "tag": tag,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": None if ttft is None else ttft * 1000.0,
        "decode_tps": decode_tps,
        "tpot_ms": tpot_ms,
        "e2e_s": t_end - t0,
        "e2e_tps": completion_tokens / (t_end - t0),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8891")
    ap.add_argument("--out", default="probe_results.json")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", default="8,32", help="comma list, e.g. 8 or 8,32")
    args = ap.parse_args()

    timeout = httpx.Timeout(600.0, connect=20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        models = (await client.get(f"{args.base_url}/v1/models")).json()
        model = models["data"][0]["id"]

        prompt = build_prompt()
        # Warmup: first request after boot carries kernel/graph warmup cost.
        warm = await one_stream(client, args.base_url, model, "Hello.", 8, "warmup")

        single = []
        for i in range(3):
            single.append(
                await one_stream(
                    client, args.base_url, model, prompt, args.max_tokens, f"single_{i + 1}"
                )
            )

        concurrency = {}
        for conc in [int(x) for x in args.concurrency.split(",") if x.strip()]:
            osl = 128
            t0 = time.perf_counter()
            runs = await asyncio.gather(
                *[
                    one_stream(client, args.base_url, model, prompt, osl, f"c{conc}_{i}")
                    for i in range(conc)
                ]
            )
            wall = time.perf_counter() - t0
            outs = [r["completion_tokens"] for r in runs]
            ttfts = [r["ttft_ms"] for r in runs if r["ttft_ms"] is not None]
            tpots = [r["tpot_ms"] for r in runs if r["tpot_ms"] is not None]
            concurrency[str(conc)] = {
                "requests": conc,
                "osl_target": osl,
                "wall_s": wall,
                "total_output_tokens": sum(outs),
                "aggregate_output_tps": sum(outs) / wall,
                "mean_ttft_ms": statistics.fmean(ttfts) if ttfts else None,
                "p50_ttft_ms": statistics.median(ttfts) if ttfts else None,
                "max_ttft_ms": max(ttfts) if ttfts else None,
                "mean_tpot_ms": statistics.fmean(tpots) if tpots else None,
                "per_request_tps": [r["decode_tps"] for r in runs],
            }

    result = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "base_url": args.base_url,
        "prompt_tokens": single[0].get("prompt_tokens") if single else warm.get("prompt_tokens"),
        "warmup": warm,
        "single_stream": single,
        "single_stream_median": {
            "ttft_ms": statistics.median([s["ttft_ms"] for s in single if s["ttft_ms"]]),
            "decode_tps": statistics.median([s["decode_tps"] for s in single if s["decode_tps"]]),
            "tpot_ms": statistics.median([s["tpot_ms"] for s in single if s["tpot_ms"]]),
        },
        "concurrency": concurrency,
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
