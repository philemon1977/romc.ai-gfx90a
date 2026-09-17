#!/usr/bin/env python3
"""Prefix-cache reuse probe: send the same long prefix twice (different question)
and compare TTFT. With prefix caching working, the second request should reuse the
cached blocks and start generating almost immediately; with prefix caching disabled
(MTP on this hybrid Mamba model, see the vLLM warning) the second request pays the
full prefill again.

Usage: prefix_reuse_probe.py [port] [model] [prefix_tokens]
"""
import json
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8100"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Ornith-640k"
TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 520000

FILLER = ("The AMD Instinct MI250X accelerator is a dual-die GPU based on the CDNA 2 "
          "architecture, providing 128 GB of HBM2e memory and roughly 3.2 TB/s of "
          "aggregate memory bandwidth. ")


def build_prefix(target_tokens: int) -> str:
    reps = max(1, target_tokens // (len(FILLER) // 4))
    return FILLER * reps


def send(prefix: str, question: str, label: str) -> dict:
    prompt = prefix + "\n\nQuestion: " + question + "\nAnswer:"
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": 16,
                       "temperature": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        d = json.load(r)
    dt = time.time() - t0
    pt = d["usage"]["prompt_tokens"]
    print(f"[{label}] prompt_tokens={pt:,} wall={dt:.1f}s  "
          f"({'reused prefix (fast)' if dt < 60 else 'FULL RE-PREFILL'})", flush=True)
    return {"label": label, "prompt_tokens": pt, "seconds": dt}


def main() -> None:
    prefix = build_prefix(TARGET)
    print(f"[probe] prefix chars={len(prefix):,} (~{len(prefix)//4:,} tokens target {TARGET:,})")
    r1 = send(prefix, "What GPU architecture is described in the text?", "req1-cold")
    r2 = send(prefix, "What memory capacity is mentioned in the text?", "req2-warm")
    ratio = r2["seconds"] / max(r1["seconds"], 1e-9)
    print(f"[probe] req2/req1 time ratio = {ratio:.3f}  "
          f"({'PREFIX REUSE WORKS' if ratio < 0.2 else 'NO REUSE (prefix cache disabled)'})")
    json.dump({"req1": r1, "req2": r2, "ratio": ratio},
              open("/work/prefix_reuse.json", "w"), indent=2)


if __name__ == "__main__":
    main()
