#!/usr/bin/env python3
"""Archive the local FP8->BF16 dequant-emulation boot evidence into the KB.

What this records (all from the 2026-09-15 session ``20260915T161838Z-ba2694ea``,
which never reached a baseline):

* the patched worktree **does** take this checkpoint's weights on vLLM/gfx90a —
  twice independently, same numbers: 48.27 GiB/die, model load 266 s / 272 s,
  GPU KV cache 154,624 tokens (25.17x concurrency at ``--max-model-len 6144``),
  then CUDA-graph capture;
* the boot recipe (which env/cache state it needs) so the next session does not
  rediscover it;
* the operational blocker: a live Hyperloom run **reaps foreign GPU processes**,
  so a side-probe cannot run alongside it — it must wait for the run to close;
* what is still unmeasured: TTFT / throughput on this route.

Idempotent: merges on the same stable key as the other seeders, and skips the
write when nothing changed.

Usage:
    PYTHONPATH=hyperloom python3 scripts/note_emulation_boot.py [--root DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "hyperloom"))

from hyperloom.orchestrator.knowledge.recipe_kb import LocalRecipeStore  # noqa: E402
from seed_reference_recipes import (  # noqa: E402
    ORNITH_CID,
    _canon,
    _now,
    merge,
)

SESSION = "20260915T161838Z-ba2694ea"
SPECIALIST = "fc07edbde4c9489682f2cafa5a7d253a"
PROBE_SPECIALIST = "8628519056b24176b3fa5c9a0e106dd6"

BOOT_RECIPE = "\n".join(
    [
        "# inside hyperloom-srv (that is where /opt/envs/vllm + ROCm live)",
        "export PYTHONPATH=<specialist run>/worktree        # patched vLLM source",
        "export VLLM_CACHE_ROOT=<fresh dir>                  # AOT graph is hash-keyed:",
        "export TORCHINDUCTOR_CACHE_DIR=<fresh dir>/inductor # a selection-only patch does NOT invalidate it",
        "export VLLM_ROCM_USE_AITER=0 HSA_NO_SCRATCH_RECLAIM=1",
        "vllm serve /mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-FP8 \\",
        "  --port <port> --tensor-parallel-size=8 --gpu-memory-utilization 0.95 \\",
        "  --max-model-len 6144 --trust-remote-code",
    ]
)

WHAT_WORKED = [
    {
        "description": (
            "反量化仿真路线在 gfx90a 上可重复地起得来（两次独立启动，数字一致）："
            "emulation.py:175 'Emulating FP8 w8a8 dense linears with bfloat16 GEMM ... dequantizing weights at load' "
            "→ 'Model loading took 48.27 GiB memory and 266 s'（首次）/ 272 s（二次）"
            "→ hybrid cache 分页对齐（attention block 528 / mamba page padding 1.34%）"
            "→ torch.compile → CUDA graph capture（51 PIECEWISE + 35 FULL）"
        ),
        "measured_impact": (
            "权重按 TP8 全部加载（48.27 GiB/die），服务容量 GPU KV cache 154,624 tokens / "
            "max concurrency 25.17x @ max-model-len 6144"
        ),
    },
    {
        "description": "起服务所需的最小配置（第二次启动实测复现）：" + BOOT_RECIPE.replace("\n", " ; "),
        "measured_impact": "缺失 fresh VLLM_CACHE_ROOT 时会重放 FP8 [K,N] 的旧 AOT 图，在 profile_run 判死",
    },
]

PITFALLS = [
    {
        "description": (
            "在跑的 Hyperloom run 会主动回收外来 GPU 进程，侧路探测无法与之共存："
            f"session {SESSION} 的 specialist {PROBE_SPECIALIST} 读到外来探测脚本后执行 "
            "'kill -TERM -35120'，把已跑到 CUDA graph capture 的独立仿真服务直接杀掉去腾卡。"
            "要独立测这份仿真路线，只能等 run 收官（或在 --resume-from 之前先测）"
        ),
        "severity": "crash",
    }
]

GAPS = [
    {
        "description": (
            "反量化仿真路线的 TTFT / 输出吞吐仍未测：本次探测在 graph capture 阶段被同机 run 回收，"
            "而 run 自己的基准又被预算跳过（grid_runner: 4293s left cannot fit 2x7800s → variant not_run）"
        ),
        "metrics": "N/A",
    }
]

EXTRAS = {
    "emulation_boot": {
        "verified_at": "2026-09-15T18:12Z",
        "weights_per_die_gib": 48.27,
        "model_load_sec": [266.1, 272.0],
        "gpu_kv_cache_tokens": 154624,
        "max_concurrency_at_6144": 25.17,
        "cuda_graphs": {"piecewise": 51, "full_decode": 35},
        "launch_recipe": BOOT_RECIPE,
        "patch": "FP8W8A8EmulationLinearKernel (dense linears dequant FP8->BF16 at load, then F.linear)",
        "measured_ttft_tps": None,
        "blocked_by": "in-flight Hyperloom run reaps foreign GPU processes",
    }
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root) if args.root else REPO / "hyperloom" / "kb"
    store = LocalRecipeStore(root=root)
    live = store._live_path(ORNITH_CID)  # noqa: SLF001
    if not live.is_file():
        raise SystemExit(f"Ornith row not found: {live}")
    row = json.loads(live.read_text())

    kwargs = {
        "canonical_id": ORNITH_CID,
        "model": row.get("model", "Ornith-1.5-397B-FP8"),
        "hardware": row.get("hardware", "mi250x"),
        "framework_name": row.get("framework_name") or row.get("framework") or "vllm",
        "framework_version": row.get("framework_version", "0.28.0"),
        "precision": row.get("precision", "fp8"),
        "best_config": row.get("best_config") or {},
        "best_throughput": float(row.get("best_throughput") or 0.0),
        "last_profiled": row.get("last_profiled", ""),
        "stack_fingerprint": row.get("stack_fingerprint") or {},
        "sessions": row.get("sessions") or [],
        "authority": row.get("authority", "EXPERIENTIAL"),
        "confidence": float(row.get("confidence") or 0.85),
        "provenance": None,
    }
    stats: dict[str, int] = {}
    for field, additions in (
        ("what_worked", WHAT_WORKED),
        ("pitfalls", PITFALLS),
        ("remaining_gaps", GAPS),
    ):
        merged, added = merge(row.get(field), additions)
        kwargs[field] = merged
        stats[field] = added

    extras = {
        k: v
        for k, v in row.items()
        if k not in {
            "canonical_id", "version", "created_at", "updated_at", "model", "hardware",
            "framework_name", "framework", "framework_version", "precision", "best_config",
            "best_throughput", "what_worked", "what_failed", "remaining_gaps", "pitfalls",
            "lessons", "last_profiled", "stack_fingerprint", "sessions", "authority",
            "confidence", "evidence_refs", "provenance", "_field_sources", "_sources",
            "prs_tested",
        }
    }
    prior_boot = row.get("emulation_boot") or {}
    merged_boot = {**prior_boot, **EXTRAS["emulation_boot"]}
    if prior_boot.get("measured_ttft_tps"):
        merged_boot["measured_ttft_tps"] = prior_boot["measured_ttft_tps"]
        merged_boot["blocked_by"] = ""
    extras["emulation_boot"] = merged_boot
    extras["emulation_note_at"] = _now()
    kwargs["extras"] = extras

    unchanged = not any(stats.values()) and _canon(prior_boot) == _canon(merged_boot)
    if unchanged:
        print(f"unchanged {ORNITH_CID} (version={row.get('version')})")
        return 0
    if args.dry_run:
        print(f"[dry-run] would update {ORNITH_CID}: added={stats}")
        return 0
    res = store.put_recipe(**kwargs)
    print(f"updated {res['canonical_id']} version={res['version']} added={stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
