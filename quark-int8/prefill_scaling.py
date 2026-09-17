#!/usr/bin/env python3
"""Characterize chunked-prefill throughput vs context length, to extrapolate the
cost of very long prompts (512K / 1M). Decode is capped at 8 tokens so the number
is dominated by prefill.
"""
import json
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8100"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Ornith-mtp"
SIZES = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3
                          else ["16384", "32768", "65536", "131072"])]

FILLER = ("The AMD Instinct MI250X accelerator is a dual-die GPU based on the CDNA 2 "
          "architecture, providing 128 GB of HBM2e memory and roughly 3.2 TB/s of "
          "aggregate memory bandwidth. ")


def build(target_tokens: int) -> str:
    unit = len(FILLER) // 4          # approx tokens per filler copy
    reps = max(1, target_tokens // unit)
    return FILLER * reps + "\n\nSummarize the text above in one short sentence.\nAnswer:"


def main() -> None:
    print(f"{'target':>8} {'actual':>9} {'time(s)':>9} {'prefill tok/s':>14}")
    for target in SIZES:
        prompt = build(target)
        body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": 8,
                           "temperature": 0, "ignore_eos": True}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=7200) as r:
            d = json.load(r)
        dt = time.time() - t0
        pt = d["usage"]["prompt_tokens"]
        print(f"{target:>8} {pt:>9,} {dt:>9.1f} {pt/dt:>14,.0f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
