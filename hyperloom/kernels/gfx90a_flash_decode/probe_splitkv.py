#!/usr/bin/env python3
"""One-config probe of the in-tree split-KV kernel (crash bisection).

Run one context length per process so a GPU memory fault (which kills the whole
process) does not take the sweep with it. Prints a single machine-readable line.

    VLLM_ROCM_SPLITKV_PA=1 [VLLM_ROCM_SPLITKV_PA_PART=512] \
      HIP_VISIBLE_DEVICES=0 python3 probe_splitkv.py <ctx>
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_in_tree_splitkv import HEAD_DIM, N_KV, N_PER_KV, build, call_in_tree  # noqa: E402

from triton.testing import do_bench  # noqa: E402


def main() -> int:
    ctx = int(sys.argv[1])
    phys = int(sys.argv[2]) if len(sys.argv) > 2 else 528
    t = build(ctx, phys)
    ok, out, st = call_in_tree(t)
    ll = st.get("last_launch") or {}
    if not ok:
        print(f"ctx={ctx} part={os.environ.get('VLLM_ROCM_SPLITKV_PA_PART','adaptive')} "
              f"tight={os.environ.get('VLLM_ROCM_SPLITKV_PA_TIGHT_GRID','1')} "
              f"REJECTED reasons={list((st.get('reject_by_reason') or {}).items())[-1:]}")
        return 3
    ms = do_bench(lambda: call_in_tree(t)[1], warmup=5, rep=20)
    gbs = 2 * ctx * N_KV * HEAD_DIM * 2 / (ms * 1e-3) / 1e9
    finite = torch_is_finite(out)
    print(f"ctx={ctx} part={os.environ.get('VLLM_ROCM_SPLITKV_PA_PART','adaptive')} "
          f"tight={os.environ.get('VLLM_ROCM_SPLITKV_PA_TIGHT_GRID','1')} "
          f"max_parts={ll.get('max_parts')} bound={ll.get('bound')} "
          f"t={ms*1000:.1f}us {gbs:.1f}GB/s finite={finite} "
          f"implied_TPOT={45.4 + 15*ms:.1f}ms")
    return 0


def torch_is_finite(x) -> bool:
    import torch

    return bool(torch.isfinite(x.float()).all().item())


if __name__ == "__main__":
    raise SystemExit(main())
