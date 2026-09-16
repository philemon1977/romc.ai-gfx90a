#!/usr/bin/env python3
"""Record the end-to-end TTFT/throughput of the FP8->BF16 emulation route.

Context: session ``20260915T161838Z-ba2694ea`` ended at 2026-09-15T19:15:59Z with
``stop_reason=enablement_stalled`` / ``baseline_tput=0.0`` (the optimizer's own
benchmark grid never ran: "4293s left cannot fit 2x7800s" -> variant not_run), so
the *only* end-to-end numbers for this route come from side probes.  Two
independent boots measured it:

* live specialist server ``:8892`` (during the run, before close);
* post-close standalone autoprobe ``:8891`` (after the run released the GPUs),
  see ``hyperloom/.tmp/emulation_autoprobe/PROBE_SUMMARY.md``.

Both agree within noise, which is the point: the route is reproducible even
though it is not an optimizer-validated baseline.

This script also repairs what the run's CLOSE sediment dropped: the
``recipe_finalize`` write at 19:16:06Z (version 9) kept ``pitfalls`` /
``remaining_gaps`` / the top-level ``vendor_reference`` + ``emulation_boot``
keys but reset ``what_worked`` (5 entries -> 0) and ``lessons``.  Those entries
are restored here from the archived ``history/v8.json`` snapshot.

Idempotent: merges on stable prefixes, drops the now-closed gap, and skips the
write when nothing changed.

Usage:
    PYTHONPATH=hyperloom python3 scripts/note_probe_crosscheck.py [--root DIR] [--dry-run]
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
    _now,
    merge,
)

SESSION = "20260915T161838Z-ba2694ea"
AUTOPROBE = REPO / "hyperloom" / ".tmp" / "emulation_autoprobe"

POST_CLOSE = {
    "tag": "post_close_autoprobe",
    "endpoint": "standalone server :8891, worktree 8628519056 (patches 001+002)",
    "measured_at": "2026-09-15T19:26:27Z",
    "isl_tokens": 776,
    "osl_tokens": 256,
    "single_stream_median": {"ttft_ms": 376.7, "decode_tps": 23.45, "tpot_ms": 42.6, "e2e_s": 11.24},
    "single_stream_cold_first_call_ms": 1175.6,
    "conc8": {
        "osl": 128,
        "aggregate_output_tps": 60.77,
        "mean_ttft_ms": 5313.8,
        "p50_ttft_ms": 6018.5,
        "mean_tpot_ms": 90.7,
        "wall_s": 16.85,
    },
    "conc32": {
        "osl": 128,
        "aggregate_output_tps": 236.74,
        "mean_ttft_ms": 2793.3,
        "p50_ttft_ms": 2700.5,
        "max_ttft_ms": 3514.8,
        "mean_tpot_ms": 113.0,
        "wall_s": 17.30,
        "per_request_tps_range": [7.74, 9.22],
    },
    "cross_boot_agreement": (
        "live :8892 vs post-close :8891 — TTFT 373 / 377 ms, decode 23.40 / 23.45 tok/s, "
        "TPOT 42.7 / 42.6 ms, conc8 aggregate 61.1 / 60.8 tok/s: two independent boots, same numbers"
    ),
    "caveat": (
        "not an optimizer sealed baseline: no warmup/sigma protocol, OSL 128 at conc>1, "
        "and this route has no MTP/ngram speculative decoding — the llama.cpp Q8_0 arm "
        "(47.28 tok/s decode) does, so the two are not like-for-like"
    ),
}

RUN_OUTCOME = {
    "session": SESSION,
    "stop_ts": "2026-09-15T19:15:59Z",
    "stop_reason": "enablement_stalled",
    "baseline_tput": 0.0,
    "cumulative_gain_validated": 0.0,
    "note": (
        "预算内没拿下 sealed baseline（grid_runner 判定 '4293s left cannot fit 2x7800s' 直接 not_run），"
        "本行 best_throughput 因此仍为 0.0：这次 run 的结论是『反量化仿真路线可加载、可起服务、可复现』，"
        "不是任何性能增益"
    ),
}

WHAT_WORKED = [
    {
        "description": (
            "反量化仿真路线端到端实测（收官后独立起服务复测，与 run 中 live server 交叉一致）："
            "单流 ISL 776 / OSL 256 → TTFT 377 ms、decode 23.45 tok/s、TPOT 42.6 ms；"
            "conc 8 → 聚合 60.8 tok/s（mean TTFT 5314 ms / TPOT 90.7 ms）；"
            "conc 32 → 聚合 236.7 tok/s（mean TTFT 2793 ms / TPOT 113.0 ms，单请求 7.7-9.2 tok/s）"
        ),
        "measured_impact": (
            "1→32 流聚合约 10x（23.5 → 236.7 tok/s）；TPOT 42.6 → 113.0 ms 退化；"
            "conc32 的 mean TTFT 反而降到 2.79 s（批次变大摊薄了 prefill）。"
            "decode 42.6 ms/token 对 48.27 GiB/die 已贴近访存roofline → 继续提速要靠更少每token字节（更低精度权重），"
            "不是kernel调优"
        ),
    },
    {
        "description": (
            "两次独立启动数字可复现（live :8892 与收官后 :8891）："
            "TTFT 373 / 377 ms、decode 23.40 / 23.45 tok/s、TPOT 42.7 / 42.6 ms、"
            "conc8 聚合 61.1 / 60.8 tok/s；KV cache 154,624 tokens = 25.17x @max-model-len 6144"
        ),
        "measured_impact": "误差在 1% 内 → 该路线虽非 sealed baseline，但测量值可作为工程口径使用",
    },
    {
        "description": (
            "复现命令（收官后独立实测通过）："
            "docker exec hyperloom-srv bash -c \"cd /home/qiba/ROCm.AI/hyperloom/.tmp/emulation_autoprobe && "
            "bash run_autoprobe.sh\" —— 脚本会等 run 收尾、按 PID 释放 GPU、用 worktree 8628519056 + 补丁 001/002 "
            "在 :8891 起服务，跑 probe.py 后自行停服务；产物 PROBE_SUMMARY.md / probe_results.json"
        ),
        "measured_impact": (
            "必须等 run 收官再测：在跑的 run 会 kill 外来 GPU 进程（此前探针于 CUDA graph capture 阶段被杀）"
        ),
    },
]

PITFALLS = [
    {
        "description": (
            "optimizer CLOSE 的 sediment（recipe_finalize）会重写本行并丢掉 what_worked / lessons："
            f"session {SESSION} 在 19:16:06Z 写入 v9，pitfalls / remaining_gaps / vendor_reference / emulation_boot 都保住了，"
            "但 what_worked 从 5 条变 0 条（已从 history/v8.json 回填）。归档后要复查这两类字段"
        ),
        "severity": "data-loss",
    }
]

GAPS = [
    {
        "description": (
            "未按 optimizer 口径复测：正式基准是 conc 64 / ISL 1024 / OSL 1024 + 多次方差，"
            "本次只有 ISL 776 / OSL 256（单流）与 OSL 128（conc 8/32）。conc 64@1k/1k 在容量上可行"
            "（单序列 2048 ≤ 6144，KV 上限约 75 序列），但聚合吞吐拐点未知"
        ),
        "metrics": "N/A",
    },
    {
        "description": (
            "本路线在 optimizer 内仍无 sealed baseline（baseline_tput=0.0、cumulative_gain_validated=0.0），"
            "best_throughput 保持 0.0：探针数字不能当作与其它臂可比的分数"
        ),
        "metrics": "baseline_tput=0.0",
    },
    {
        "description": (
            "未测投机解码（MTP / ngram）在本路线的收益：llama.cpp Q8_0 臂在同机 47.28 tok/s 带 ngram+MTP，"
            "要判断仿真臂差距必须先把这两条口径对齐（或给 vLLM 侧也开投机）"
        ),
        "metrics": "N/A",
    },
]

# the gap this evidence closes
CLOSED_GAP_MARKER = "TTFT / 输出吞吐仍未测"

# run-side failure events arrive as bare reasons with an empty description;
# fill them from what this session actually established was behind each reason.
FAILURE_NOTES = {
    "subprocess_nonzero": (
        "fp8 stock 路线：vllm serve 子进程非零退出 —— 首个 fp8 linear apply 就撞 "
        "torch._scaled_mm 引擎级门（'only supported on CUDA devices with compute capability "
        ">= 9.0 or 8.9, or ROCm MI300+'），gfx90a 无 FP8 矩阵核"
    ),
    "server_init_dead": (
        "/v1/models 未 ready：服务死在 profile_run / CUDA graph capture（旧 FP8 [K,N] AOT 图重放 "
        "触发 assert_size_stride），本 run 内三次 server_init_dead"
    ),
}


def _enrich_failures(entries: list | None) -> tuple[list, int]:
    """Give the run's bare ``reason``-only failure rows a description, de-duplicated."""
    out: list = []
    by_reason: dict[str, dict] = {}
    filled = 0
    for entry in entries or []:
        if not isinstance(entry, dict) or entry.get("description"):
            out.append(entry)
            continue
        reason = str(entry.get("reason") or "")
        if reason in by_reason:
            by_reason[reason]["occurrences"] = int(by_reason[reason].get("occurrences", 1)) + 1
            continue
        note = {
            "description": FAILURE_NOTES.get(reason, f"运行期失败（reason={reason}，无详情）"),
            "reason": reason,
            "occurrences": 1,
        }
        by_reason[reason] = note
        out.append(note)
        filled += 1
    return out, filled


def _from_history(hist: Path, field: str) -> list:
    """Newest archived snapshot value for ``field`` (empty when absent)."""
    if not hist.is_dir():
        return []
    for version in sorted(
        (int(p.stem[1:]) for p in hist.glob("v*.json") if p.stem[1:].isdigit()), reverse=True
    ):
        snap = (json.loads((hist / f"v{version}.json").read_text()) or {}).get("snapshot") or {}
        if snap.get(field):
            return list(snap[field])
    return []



def _drop_closed_gaps(gaps: list | None) -> tuple[list, int]:
    out, dropped = [], 0
    for entry in gaps or []:
        blob = json.dumps(entry, ensure_ascii=False) if isinstance(entry, dict) else str(entry)
        if CLOSED_GAP_MARKER in blob:
            dropped += 1
            continue
        out.append(entry)
    return out, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root) if args.root else REPO / "hyperloom" / "kb"
    store = LocalRecipeStore(root=root)
    live_path = store._live_path(ORNITH_CID)  # noqa: SLF001
    if not live_path.is_file():
        raise SystemExit(f"Ornith row not found: {live_path}")
    row = json.loads(live_path.read_text())

    kwargs = {
        "canonical_id": ORNITH_CID,
        "model": row.get("model", "Ornith-1.5-397B-FP8"),
        "hardware": row.get("hardware", "mi250x"),
        "framework_name": row.get("framework_name") or row.get("framework") or "vllm",
        "framework_version": row.get("framework_version", "0.28.0"),
        "precision": row.get("precision", "fp8"),
        "best_config": row.get("best_config") or {},
        # stays 0.0 on purpose: no optimizer-sealed baseline was ever taken
        "best_throughput": float(row.get("best_throughput") or 0.0),
        "last_profiled": POST_CLOSE["measured_at"],
        "stack_fingerprint": row.get("stack_fingerprint") or {},
        "sessions": row.get("sessions") or [],
        "authority": row.get("authority", "EXPERIENTIAL"),
        "confidence": float(row.get("confidence") or 0.85),
        "provenance": None,
    }
    STATUS_NOTE = (
        "原判 dead 针对无补丁的 stock vLLM fp8 路线；反量化仿真内核（补丁 001+002）已证明"
        "可加载、可起服务、可复现。收官后独立复测：单流 TTFT 377 ms / decode 23.45 tok/s，"
        "conc32 聚合 236.7 tok/s。但仍无 optimizer sealed baseline（baseline_tput=0.0），"
        "本行仍是 EXPERIENTIAL，best_throughput 保持 0.0"
    )

    # put_recipe writes exactly the fields it is handed: anything a previous
    # writer (e.g. the run's CLOSE sediment) dropped comes back empty, so every
    # list field is carried over explicitly, restoring from history when the
    # live row lost it.
    stats: dict[str, int] = {}
    carried: dict[str, list] = {}
    for field in ("what_worked", "what_failed", "pitfalls", "lessons"):
        live_val = list(row.get(field) or [])
        if not live_val:
            live_val = _from_history(live_path.parent / "history", field)
            if live_val:
                stats[f"{field}_restored"] = len(live_val)
        carried[field] = live_val
    for field, additions in (("what_worked", WHAT_WORKED), ("pitfalls", PITFALLS)):
        merged, added = merge(carried[field], additions)
        kwargs[field] = merged
        stats[field] = added
    failures, filled = _enrich_failures(carried["what_failed"])
    kwargs["what_failed"] = failures
    if filled:
        stats["what_failed_enriched"] = filled
    kwargs["lessons"] = carried["lessons"]

    gaps, dropped = _drop_closed_gaps(row.get("remaining_gaps"))
    gaps, added = merge(gaps, GAPS)
    kwargs["remaining_gaps"] = gaps
    stats["remaining_gaps"] = added
    stats["remaining_gaps_closed"] = dropped

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
    boot = dict(row.get("emulation_boot") or {})
    boot["blocked_by"] = ""
    boot["crosscheck_at"] = POST_CLOSE["measured_at"]
    boot["post_close_crosscheck"] = POST_CLOSE
    boot["run_outcome"] = RUN_OUTCOME
    boot["artifacts"] = [
        "hyperloom/reports/fp8-emulation-probe/PROBE_SUMMARY.md",
        "hyperloom/reports/fp8-emulation-probe/probe_results.json",
        "hyperloom/reports/fp8-emulation-probe/autoprobe.log",
        "hyperloom/reports/fp8-emulation-probe/run_autoprobe.sh",
        "hyperloom/.tmp/emulation_autoprobe/  (working origin)",
        "hyperloom/session/Ornith-1.5-397B-FP8/20260915T161838Z-ba2694ea/reports/final.md",
    ]
    extras["status"] = "experimental"
    extras["status_note"] = STATUS_NOTE
    extras["emulation_boot"] = boot
    extras["emulation_note_at"] = _now()
    kwargs["extras"] = extras

    prior_boot = row.get("emulation_boot") or {}
    added = any(stats.get(k) for k in ("what_worked", "pitfalls", "remaining_gaps"))
    recovered = any(k.endswith(("_restored", "_enriched")) for k in stats)
    unchanged = (
        bool(prior_boot.get("post_close_crosscheck"))
        and not added
        and not recovered
        and not dropped
        and (prior_boot.get("artifacts") or []) == boot["artifacts"]
        and row.get("status_note") == STATUS_NOTE
    )
    if unchanged:
        print(f"unchanged {ORNITH_CID} (version={row.get('version')})")
        return 0
    if args.dry_run:
        print(f"[dry-run] would update {ORNITH_CID}: {stats}")
        return 0
    res = store.put_recipe(**kwargs)
    print(f"updated {res['canonical_id']} version={res['version']} {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
