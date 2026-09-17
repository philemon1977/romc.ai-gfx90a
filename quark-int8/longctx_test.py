#!/usr/bin/env python3
"""Long-context functional test: build a long synthetic prompt with a needle in
the middle, ask the model to reproduce it, and measure prefill/decode timing.

Usage: longctx_test.py [port] [model] [target_tokens]
"""
import json
import os
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8100"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Ornith-mtp"
TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 32768
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 512

FILLER = ("The AMD Instinct MI250X accelerator is a dual-die GPU based on the CDNA 2 "
          "architecture, providing 128 GB of HBM2e memory per package and roughly "
          "3.2 TB/s of aggregate memory bandwidth. Each GCD exposes 110 compute units. ")
NEEDLE = "The secret passphrase is ORNITH-256K-QUANT-OK."


def approx_tokens(text: str) -> int:
    # ~4 chars per token for English text
    return len(text) // 4


def build_prompt(target_tokens: int) -> str:
    body = []
    while approx_tokens("".join(body)) < target_tokens * 0.5:
        body.append(FILLER)
    body.append("\n" + NEEDLE + "\n")
    while approx_tokens("".join(body)) < target_tokens:
        body.append(FILLER)
    body.append("\n\nQuestion: What is the secret passphrase mentioned in the text above? "
                "Answer with the passphrase only.\nAnswer:")
    return "".join(body)


def main() -> None:
    prompt = build_prompt(TARGET)
    est = approx_tokens(prompt)
    print(f"[longctx] prompt chars={len(prompt):,} (~{est:,} tokens), target={TARGET:,}")
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": MAXTOK,
                       "temperature": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        print(f"[longctx] HTTP {e.code}: {e.read().decode()[:400]}")
        raise
    dt = time.time() - t0
    pt = d["usage"]["prompt_tokens"]
    ct = d["usage"]["completion_tokens"]
    text = d["choices"][0]["text"]
    print(f"[longctx] prefill+decode {dt:.1f}s for {pt:,} prompt tokens "
          f"(~{pt/max(dt-ct/40,1e-9):,.0f} tok/s prefill, incl. {ct} decode tokens)")
    print(f"[longctx] needle found: {'YES' if 'ORNITH-256K-QUANT-OK' in text else 'NO'}")
    print(f"[longctx] answer: {text[-300:]!r}")


if __name__ == "__main__":
    main()
