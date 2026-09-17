#!/usr/bin/env python3
"""Clean per-workload step decomposition for an MTP (spec-decode) served model.

For ONE request, using the /metrics counter deltas:
    steps      = Δdraft_tokens / SPEC          (每步投 SPEC 个草稿)
    tok/step   = completion_tokens / steps     (= 1 + 接受个数)
    step_ms    = wall_time / steps
This separates "the arm is slower per step" from "the arm accepts fewer draft
tokens" — the two causes of a TPS gap that a raw tok/s number cannot tell apart.

Usage: step_probe.py PORT TAG SPEC [mode=count|explain] [maxtok=256]
"""
import json
import os
import re
import statistics
import sys
import time
import urllib.request

PORT = sys.argv[1]
TAG = sys.argv[2]
SPEC = int(sys.argv[3])
MODE = sys.argv[4] if len(sys.argv) > 4 else "count"
MAXTOK = int(sys.argv[5]) if len(sys.argv) > 5 else 256

EXPLAIN = ("Explain in detail why int4 quantization reduces memory bandwidth pressure "
           "during decoding, covering weights and KV cache.")
COUNT = "Count slowly from one to ninety, writing each number in words on its own line."
PROMPT = COUNT if MODE == "count" else EXPLAIN
MODEL = os.environ.get("MODEL", "ornith")   # 2026-09-18：原硬编码 ornith ⇒ 换模型必 404


def counters():
    txt = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=60).read().decode()
    out = {}
    for key in ("spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total"):
        m = re.search(rf"^(?:vllm:)?{key}\{{[^}}]*\}}\s+([0-9.eE+]+)$", txt, re.M)
        out[key] = float(m.group(1)) if m else 0.0
    return out


def one():
    body = json.dumps({'model': MODEL, 'prompt': PROMPT, 'max_tokens': MAXTOK,
                       'temperature': 0, 'ignore_eos': True}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    return time.time() - t0, d['usage']['completion_tokens']


one()  # warmup, discarded
rows = []
for _ in range(3):
    c0 = counters()
    dt, ntok = one()
    c1 = counters()
    dd = c1['spec_decode_num_draft_tokens_total'] - c0['spec_decode_num_draft_tokens_total']
    da = c1['spec_decode_num_accepted_tokens_total'] - c0['spec_decode_num_accepted_tokens_total']
    steps = dd / SPEC
    rows.append((dt, ntok, dd, da, steps, ntok / steps, 1000 * dt / steps, ntok / dt))

print(f"[{TAG}] mode={MODE} spec={SPEC} maxtok={MAXTOK}")
for i, (dt, ntok, dd, da, steps, tps_step, step_ms, tps) in enumerate(rows, 1):
    print(f"  rep{i}: {ntok} tok in {dt:.2f}s = {tps:.2f} tok/s | steps {steps:.0f} "
          f"| {tps_step:.2f} tok/step (accept {100*da/dd:.1f}%) | step {step_ms:.1f} ms")
m = lambda k: statistics.median([r[k] for r in rows])
print(f"[{TAG}] MEDIAN {m(7):.2f} tok/s | {m(5):.2f} tok/step (accept "
      f"{100*sum(r[3] for r in rows)/sum(r[2] for r in rows):.1f}%) | step {m(6):.1f} ms (n=3)")
