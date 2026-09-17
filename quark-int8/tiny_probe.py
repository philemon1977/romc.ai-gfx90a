#!/usr/bin/env python3
"""Mechanics + numerical-equivalence test for the expert-cache V2 on the tiny
Qwen3_5Moe INT8 checkpoint (same architecture/quant format, 4 layers x 16 experts).

Queries /v1/completions with prompt_logprobs and prints the mean token NLL and a
hash of the first tokens, so two servers (cache on/off) can be compared exactly.
"""
import hashlib
import json
import sys
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8125"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "run"
PROMPT = ("Quantization reduces memory traffic during decoding. "
          "Mixture-of-experts layers route each token to a few experts. ")


def main() -> None:
    body = json.dumps({"model": "tiny", "prompt": PROMPT, "max_tokens": 24,
                       "temperature": 0, "ignore_eos": True,
                       "prompt_logprobs": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    text = d["choices"][0]["text"]
    pl = d["choices"][0].get("prompt_logprobs") or []
    lps = [next(iter(e.values()))["logprob"] for e in pl if e]
    nll = -sum(lps) / len(lps) if lps else float("nan")
    # the generated text is the strongest signal: greedy decoding must be identical
    h = hashlib.sha256(text.encode()).hexdigest()[:16]
    print(f"[{LABEL}] prompt_logprobs={len(lps)} mean_nll={nll:.6f} "
          f"gen_text_sha256={h} text={text[:60]!r}")
    json.dump({"label": LABEL, "mean_nll": nll, "sha": h, "text": text},
              open(f"/work/tiny_{LABEL}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
