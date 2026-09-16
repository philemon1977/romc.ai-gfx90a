#!/usr/bin/env python3
"""E: measure single-stream decode speed and MTP acceptance on REALISTIC content.

Why this exists: on the official synthetic workload (`--dataset-name random` = random
token ids, `--ignore-eos`, greedy), Ornith-1.5-397B-FP8's MTP draft is accepted
almost perfectly -- across config B: weighted acceptance 88.4%, mean acceptance
length 2.72 of a possible 3.00 at k=2, and 18/67 windows fully saturated at 3.00.
A degenerate greedy continuation of random tokens is trivially predictable, so the
resulting 2.00x (ISL 1k) / 2.61x (ISL 32k) speedups are an UPPER BOUND that will not
transfer to agent traffic, where per-step cost is the same but acceptance is what
multiplies it:

    tok/s(speedup) ~= mean_acceptance_length / per_step_cost_ratio

This probe instead feeds real Python source from this repo and asks for a
continuation, at temperature 0 -- the closest cheap stand-in for a coding-agent
turn -- and reports:
  * decode tok/s per request (same server, comparable to the synthetic points)
  * TTFT per request
  * the SpecDecoding windows the SERVER logged during the probe only (log offset
    delta), aggregated into acceptance rate + mean acceptance length

Run inside hyperloom-srv:
    /opt/envs/vllm/bin/python3 realcode_probe.py --base-url http://127.0.0.1:8901 \
        --model /mnt/.../Ornith-1.5-397B-FP8 --server-log <cfg>/server.log \
        --out <cfg>/realcode_probe.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from pathlib import Path

import httpx

REPO = Path("/home/qiba/ROCm.AI/hyperloom/hyperloom")
ACC_RE = re.compile(
    r"Mean acceptance length: ([\d.]+).*?Per-position acceptance rate: ([\d.]+), ([\d.]+)"
    r".*?Accepted: (\d+) tokens, Drafted: (\d+) tokens"
)
INSTR = "\nExplain what this module does, then list the three riskiest defects in it with a one-line fix for each.\n"


def build_prompt(item_file: str, body: str) -> list[dict]:
    """A realistic coding-agent turn: system role + repo file + a task."""
    return [
        {
            "role": "system",
            "content": "You are a senior engineer reviewing a Python repository during an autonomous coding session. Be concrete and terse.",
        },
        {"role": "user", "content": f"File: {item_file}\n```python\n{body}\n```\n{INSTR}"},
    ]


def pick_prompts(count: int, context_tokens: int) -> list[dict]:
    """Real source files, truncated to ~context_tokens, as continuation prompts."""
    files = sorted(REPO.rglob("*.py"))
    files = [f for f in files if 4000 < f.stat().st_size < 400_000]
    # spread the sample across the tree instead of taking alphabetical neighbours
    step = max(1, len(files) // max(1, count * 6))
    chosen, seen = [], set()
    for f in files[::step]:
        if f.name in seen or f.parent == REPO:
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        body = text[: context_tokens * 4]
        if len(body) < 2000:
            continue
        seen.add(f.name)
        chosen.append(
            {
                "tag": f"real_{f.name}",
                "path": str(f),
                "prompt": f"File: {f.relative_to(REPO.parent)}\n```python\n{body}\n```\n{INSTR}",
                "messages": build_prompt(str(f.relative_to(REPO.parent)), body),
            }
        )
        if len(chosen) >= count:
            break
    return chosen


async def one(client: httpx.AsyncClient, base_url: str, model: str, item: dict, max_tokens: int,
              chat: bool = True) -> dict:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": False,  # realistic: a real turn stops when it stops
    }
    if chat:
        payload["messages"] = item["messages"]
        endpoint = "/v1/chat/completions"
    else:
        payload["prompt"] = item["prompt"]
        endpoint = "/v1/completions"
    payload = {k: v for k, v in payload.items() if v is not None}
    t0 = time.perf_counter()
    ttft = first = last = None
    n = 0
    usage: dict = {}
    async with client.stream("POST", f"{base_url}{endpoint}", json=payload) as resp:
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
            for ch in chunk.get("choices") or []:
                piece = ch.get("text")
                if piece is None:
                    piece = (ch.get("delta") or {}).get("content")
                if not piece:
                    continue
                now = time.perf_counter()
                if ttft is None:
                    ttft, first = now - t0, now
                last = now
                n += 1
    end = time.perf_counter()
    dec = (last - first) if first and last and last > first else None
    # decode_tps MUST come from usage.completion_tokens, not the chunk count: under
    # speculative decoding vLLM streams several accepted tokens per chunk, so a
    # chunk-rate "tps" understates the win by roughly the acceptance length (this
    # bug made config B's MTP turns read 5.6 tok/s when e2e said 2-3x faster).
    comp_tokens = usage.get("completion_tokens") or n
    out = {
        "chunks_seen": n,
        "tag": item["tag"],
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": comp_tokens,
        "ttft_ms": round(ttft * 1000, 1) if ttft else None,
        "decode_tps": round((comp_tokens - 1) / dec, 2) if dec and comp_tokens > 1 else None,
        "chunk_rate_cps": round((n - 1) / dec, 2) if dec and n > 1 else None,
        "tokens_per_chunk": round(comp_tokens / n, 2) if n else None,
        "e2e_s": round(end - t0, 2),
    }
    print(json.dumps(out), flush=True)
    return out


def acceptance_from_log(path: Path, offset: int) -> tuple[dict, int]:
    """Aggregate SpecDecoding windows appended while the probe ran."""
    try:
        size = path.stat().st_size
        with path.open(errors="ignore") as f:
            f.seek(offset)
            tail = f.read()
    except OSError:
        return {}, offset
    acc = dra = 0
    lengths: list[float] = []
    p1: list[float] = []
    p2: list[float] = []
    for line in tail.splitlines():
        if "SpecDecoding metrics" not in line:
            continue
        m = ACC_RE.search(line)
        if not m:
            continue
        lengths.append(float(m.group(1)))
        p1.append(float(m.group(2)))
        p2.append(float(m.group(3)))
        acc += int(m.group(4))
        dra += int(m.group(5))
    if not dra:
        return {"windows": 0}, size
    return {
        "windows": len(lengths),
        "drafted_tokens": dra,
        "accepted_tokens": acc,
        "acceptance_rate_pct": round(acc / dra * 100, 1),
        "mean_acceptance_length": round(sum(lengths) / len(lengths), 3),
        "per_position": [round(sum(p1) / len(p1), 3), round(sum(p2) / len(p2), 3)] if p1 else None,
    }, size


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-log", default="")
    ap.add_argument("--requests", type=int, default=12)
    ap.add_argument("--context-tokens", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--no-chat", action="store_true", help="use /v1/completions continuation")
    args = ap.parse_args()

    items = pick_prompts(args.requests, args.context_tokens)
    if not items:
        print("no real source files found to probe with")
        return 1
    log_path = Path(args.server_log) if args.server_log else None
    offset = log_path.stat().st_size if (log_path and log_path.is_file()) else 0

    rows = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=30.0)) as client:
        for it in items:
            rows.append(await one(client, args.base_url, args.model, it, args.max_tokens,
                                   chat=not args.no_chat))

    acc, _ = acceptance_from_log(log_path, offset) if log_path else ({}, 0)
    decodes = [r["decode_tps"] for r in rows if r["decode_tps"]]
    ttfts = [r["ttft_ms"] for r in rows if r["ttft_ms"]]
    summary = {
        "content": ("real repo python source via /v1/chat/completions agent turn, "
                    "temperature 0, natural stop") if not args.no_chat else
                   "real repo python source via /v1/completions continuation, temperature 0",
        "requests": len(rows),
        "prompt_tokens_median": statistics.median([r["prompt_tokens"] for r in rows if r["prompt_tokens"]]),
        "completion_tokens_median": statistics.median(
            [r["completion_tokens"] for r in rows if r["completion_tokens"]]
        ),
        "decode_tps_median": round(statistics.median(decodes), 2) if decodes else None,
        "ttft_ms_median": round(statistics.median(ttfts), 1) if ttfts else None,
        "speculative_acceptance": acc or None,
        "per_request": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_request"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
