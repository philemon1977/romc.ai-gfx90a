#!/usr/bin/env python3
"""Single-stream TPS with repetitions, with an explicit measurement mode.

Measured 2026-09-17: with a FIXED prompt the run-to-run spread is only 1-1.6%, while
salting the prompt (different content -> different MTP acceptance) moves the median by up
to 24%. So:
  * mode=explain  -> the historical prompt; a LOW-predictability workload (~56 t/s)
  * mode=count    -> highly predictable (~86 t/s); best SNR for A/B (spread ~1.5%)
  * mode=salt     -> random salt each rep; reproduces the old practice (only good to ~10%)
The FIRST request after model load is always a warmup artifact (JIT/cache) and is reported
separately and excluded from the median.

usage: measure_median.py PORT TAG [REPS] [MAXTOK] [mode]
"""
import json
import secrets
import statistics
import sys
import time
import urllib.request

port = sys.argv[1]
tag = sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
maxtok = int(sys.argv[4]) if len(sys.argv) > 4 else 256
mode = sys.argv[5] if len(sys.argv) > 5 else "count"

EXPLAIN = ("Explain in detail why int4 quantization reduces memory bandwidth pressure "
           "during decoding, covering weights and KV cache.")
COUNT = "Count slowly from one to ninety, writing each number in words on its own line."


def prompt_for(rep: int) -> str:
    if mode == "salt":
        return f"[salt {secrets.token_hex(4)}] {EXPLAIN}"
    if mode == "explain":
        return EXPLAIN
    return COUNT


def one(prompt: str) -> float:
    body = json.dumps({'model': 'ornith', 'prompt': prompt, 'max_tokens': maxtok,
                       'temperature': 0, 'ignore_eos': True}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    dt = time.time() - t0
    u = d['usage']
    return u['completion_tokens'] / dt


warm = one(prompt_for(0))
print(f"  [{tag}] warmup(first request, excluded): {warm:.2f} tok/s", flush=True)
tps = []
for rep in range(reps):
    r = one(prompt_for(rep + 1))
    tps.append(r)
    print(f"  [{tag}] rep{rep+1}: {r:.2f} tok/s", flush=True)
med = statistics.median(tps)
print(f"[{tag}] mode={mode} MEDIAN {med:.2f} tok/s  min {min(tps):.2f}  max {max(tps):.2f}  "
      f"spread {100*(max(tps)-min(tps))/med:.1f}%  (n={reps}, first request excluded)")
