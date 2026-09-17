#!/usr/bin/env python3
"""Concurrency benchmark: fire N simultaneous streaming requests and report
aggregate throughput, TTFT and per-request latency.

Usage: bench_concurrency.py [port] [model] [levels] [prompt_tokens] [max_tokens]
       levels default "1,4,8,16"
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8100"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Ornith-1.5-397B-Int8-512K"
LEVELS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,4,8,16").split(",")]
PROMPT_TOK = int(sys.argv[4]) if len(sys.argv) > 4 else 400
MAX_TOK = int(sys.argv[5]) if len(sys.argv) > 5 else 128

FILLER = ("The AMD Instinct MI250X accelerator uses CDNA 2 compute units and HBM2e memory. ")


def build_prompt(n_tokens: int) -> str:
    reps = max(1, n_tokens // (len(FILLER) // 4))
    return FILLER * reps + "\n\nSummarize the text above in one short sentence.\nAnswer:"


def one_request(prompt: str) -> dict:
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": MAX_TOK,
                       "temperature": 0, "ignore_eos": True, "stream": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    ntok = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except Exception:
                continue
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                if ttft is None:
                    ttft = time.time() - t0
                ntok += 1
    return {"ttft": ttft or float("nan"), "total": time.time() - t0, "tokens": ntok}


def level(concurrency: int, prompt: str) -> dict:
    results: list[dict] = []
    lock = threading.Lock()

    def worker():
        r = one_request(prompt)
        with lock:
            results.append(r)

    t0 = time.time()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    total_tok = sum(r["tokens"] for r in results)
    return {
        "concurrency": concurrency,
        "wall_s": wall,
        "total_tokens": total_tok,
        "aggregate_tok_s": total_tok / wall if wall else 0.0,
        "mean_ttft_s": statistics.mean([r["ttft"] for r in results]),
        "mean_latency_s": statistics.mean([r["total"] for r in results]),
        "tokens_per_req": statistics.mean([r["tokens"] for r in results]),
    }


def main() -> None:
    prompt = build_prompt(PROMPT_TOK)
    print(f"[bench] model={MODEL} prompt~{PROMPT_TOK} tok, max_tokens={MAX_TOK}")
    rows = []
    for c in LEVELS:
        r = level(c, prompt)
        rows.append(r)
        print(f"[bench] conc={r['concurrency']:>3}  aggregate={r['aggregate_tok_s']:7.2f} tok/s  "
              f"TTFT={r['mean_ttft_s']:6.2f}s  latency/req={r['mean_latency_s']:7.2f}s  "
              f"wall={r['wall_s']:7.2f}s", flush=True)
    json.dump(rows, open("/work/concurrency_bench.json", "w"), indent=2)


if __name__ == "__main__":
    main()
