#!/usr/bin/env python3
"""CPU-only probe: what int4-family assets does aiter ship, and what does vLLM's
ROCm kernel selection think about them on gfx90a (no GPU touched)."""
import glob
import os
import re
from collections import Counter

import aiter

root = os.path.dirname(aiter.__file__)
print("aiter version:", getattr(aiter, "__version__", "n/a"), "| root:", root)

hits = {}
for dirpath, _, files in os.walk(os.path.join(root, "ops")):
    for fn in files:
        if not fn.endswith(".py"):
            continue
        s = open(os.path.join(dirpath, fn), errors="ignore").read()
        fns = sorted(set(re.findall(r"^def (\w*(?:a16w4|a4w4|mxfp4|int4)\w*)", s, re.M)))
        if fns:
            hits[fn] = fns
print("\n== aiter/ops int4-family functions ==")
for k, v in sorted(hits.items()):
    print(" ", k, v[:6])

jdir = os.path.join(root, "jit")
print("\n== aiter/jit entries matching int4 family ==")
print([m for m in os.listdir(jdir) if re.search(r"a16w4|a4w4|mxfp4|int4", m)])

meta = "/usr/local/lib/python3.12/dist-packages/aiter_meta"
cand = []
for pat in ("*a4w4*", "*a16w4*", "*mxfp4*", "*gemm_a4*"):
    cand += glob.glob(os.path.join(meta, "**", pat), recursive=True)
print(f"\n== aiter_meta sources ({len(cand)}) ==")
for c in cand[:14]:
    print(" ", c.replace(meta, ""))

cnt = Counter()
for c in cand:
    if os.path.isfile(c):
        for g in re.findall(r"gfx9[0-9a-f]+", open(c, errors="ignore").read()):
            cnt[g] += 1
print("\n== arch strings in those sources ==", dict(cnt))

# does the a8w8 module (already JIT-built for gfx90a here) have a 4-bit sibling?
try:
    import torch

    from vllm.model_executor.kernels.linear import _POSSIBLE_MXFP4_KERNELS
    from vllm.platforms import current_platform

    print("\n== vLLM MXFP4 kernels on", current_platform._enum, "==")
    for k in _POSSIBLE_MXFP4_KERNELS[current_platform._enum]:
        try:
            print(" ", k.__name__, "->", k.is_supported())
        except Exception as e:
            print(" ", k.__name__, "raise", type(e).__name__, str(e)[:90])
    from vllm.model_executor.kernels.linear.mixed_precision import (
        _POSSIBLE_KERNELS as P2,
    )

    print("== vLLM W4A16 (mixed_precision) kernels on ROCm ==")
    for k in P2[current_platform._enum]:
        try:
            print(" ", k.__name__, "->", k.is_supported() if hasattr(k, "is_supported") else "n/a")
        except Exception as e:
            print(" ", k.__name__, "raise", type(e).__name__, str(e)[:90])
except Exception as e:
    print("vLLM probe skipped:", type(e).__name__, str(e)[:120])
