#!/usr/bin/env python3
"""Seed the Hyperloom RecipeKB from /home/qiba/ai/docs/recipes extractions.

Reads the structured extractions produced from the MI250X workstation's
recipe library (``docs/recipes/serving/*.md`` → ``.tmp/kb_extract/serving_*.json``)
and writes one recipe row per serving arm into the workspace-local KB using
Hyperloom's own ``LocalRecipeStore.put_recipe`` (so the on-disk shape is
byte-compatible with what the optimizer produces at CLOSE).

KB root: ``$KNOWLEDGE_LOCAL_ROOT`` if set, else ``<repo>/hyperloom/kb``.
Identity: ``inference:{model}:{hardware}:{framework}:{model_type}:{arch}:{fw_version}:{precision}``
with hardware pinned to ``mi250x`` (single-node → gpu_type passes through
unchanged per ``kb_hardware_slug``).

Usage:
    PYTHONPATH=hyperloom python3 scripts/seed_recipe_kb.py [--root DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "hyperloom"))

from hyperloom.orchestrator.knowledge.recipe_kb import LocalRecipeStore  # noqa: E402
from hyperloom.inference_optimizer.recipe_snapshot_constants import (  # noqa: E402
    recipe_canonical_id,
)

EXTRACT_DIR = REPO / ".tmp" / "kb_extract"
SERVING_FILES = ("serving_A.json", "serving_B.json", "serving_C.json")
HARDWARE = "mi250x"
FRAMEWORK_NAMES = {"vllm": "vllm", "llamacpp": "llama.cpp", "llama.cpp": "llama.cpp"}
CONFIDENCE = {"active": 0.85, "experimental": 0.6, "archived": 0.4, "dead": 0.35}

# Host-truth framework_version per environment (dist-info / `--version`
# measured on this machine, 2026-09-15). Wins over whatever the extraction
# guessed; unknown envs keep the extracted value.
VERSION_BY_ENV = {
    "vllm_master_rocm724": "0.1.dev1+g8a728663c.rocm724",
    "vllm_0.28.0_rocm72": "0.28.0+rocm723",
    "wu1w-int8-028": "0.28.0+rocm723",
    "glm5next-rp": "0.3.0-dev+629b50552",
    "qwen4exp-mtp-rp": "0.3.0-dev+a9e9c3c5f",
    "llama.cpp_dsv41-gfx90a-2026.9.12": "0.4.0-dev+f37da57",
}


def normalize_version(arm: dict) -> str:
    env = _s(arm.get("env_path"))
    for key, ver in VERSION_BY_ENV.items():
        if key in env:
            return ver
    return _s(arm.get("framework_version"))


def _s(text: str) -> str:
    return str(text or "").strip()


def row_from_arm(arm: dict) -> dict:
    """Turn one extracted serving arm into put_recipe kwargs."""
    model = _s(arm.get("model_slug") or arm.get("model_dir"))
    framework = FRAMEWORK_NAMES.get(_s(arm.get("engine")).lower(), _s(arm.get("engine")))
    cid = recipe_canonical_id(
        model=model,
        hardware=HARDWARE,
        framework_name=framework,
        model_type=_s(arm.get("config_model_type")),
        architectures=arm.get("architectures") or [],
        framework_version=normalize_version(arm),
        precision=_s(arm.get("precision")),
    )
    best_args = " ".join(str(t) for t in arm.get("best_args") or [])
    best_env = {str(k): str(v) for k, v in (arm.get("best_env") or {}).items()}
    primary = arm.get("primary_metric") or {}
    try:
        best_throughput = float(primary.get("value") or 0.0)
    except (TypeError, ValueError):
        best_throughput = 0.0

    verified = _s(arm.get("verified"))
    note_bits = [
        f"port {arm.get('port')}",
        f"status {arm.get('status')}",
        f"gpus {arm.get('gpus')}",
        f"TP {arm.get('tp')}",
    ]
    session = {
        "date": verified,
        "throughput_before": 0.0,
        "throughput_after": best_throughput,
        "actions_taken": [
            f"seeded from docs/recipes/{arm.get('recipe_id')}",
        ],
        "session_id": f"recipe-seed:{arm.get('recipe_id')}",
        "gain_pct": 0.0,
        "stack_len": len(arm.get("patches") or []),
    }
    fp = arm.get("stack_fingerprint") or {}
    kwargs = dict(
        canonical_id=cid,
        model=model,
        hardware=HARDWARE,
        framework_name=framework,
        framework_version=normalize_version(arm),
        precision=_s(arm.get("precision")),
        best_config={"extra_server_args": best_args, "extra_envs": best_env},
        best_throughput=best_throughput,
        what_worked=arm.get("what_worked") or [],
        what_failed=arm.get("what_failed") or [],
        remaining_gaps=arm.get("remaining_gaps") or [],
        pitfalls=arm.get("pitfalls") or [],
        lessons=arm.get("lessons") or [],
        last_profiled=verified,
        stack_fingerprint={
            "vllm_version": _s(fp.get("vllm_version")),
            "aiter_commit": _s(fp.get("aiter_commit")),
            "rocm_version": _s(fp.get("rocm_version")) or "7.2.4",
        },
        sessions=[session],
        authority="EXPERIENTIAL",
        confidence=CONFIDENCE.get(_s(arm.get("status")), 0.85),
        evidence_refs=[
            f"/home/qiba/ai/docs/recipes/serving/{arm.get('recipe_id')}.md",
            *(_s(arm.get("entry")) and [f"/home/qiba/ai/{_s(arm.get('entry'))}"] or []),
            *[f"/home/qiba/ai/{p}" for p in (arm.get("sources") or []) if _s(p).startswith(("docs/", "models/", "config/"))],
        ],
        provenance={
            "details": (
                "Seeded from the workstation recipe library docs/recipes/ "
                f"(verified {verified or 'unknown'}); metric: {primary.get('phase', '')} "
                f"{primary.get('value', '')}{primary.get('unit', '')} — {primary.get('note', '')}"
            ),
            "seed_script": "scripts/seed_recipe_kb.py",
            "recipe_id": _s(arm.get("recipe_id")),
        },
        extras={
            "recipe_id": _s(arm.get("recipe_id")),
            "port": _s(arm.get("port")),
            "engine": _s(arm.get("engine")),
            "status": _s(arm.get("status")),
            "tp": int(arm.get("tp") or 0),
            "gpus": _s(arm.get("gpus")),
            "entry": _s(arm.get("entry")),
            "model_dir": _s(arm.get("model_dir")),
            "env_path": _s(arm.get("env_path")),
            "patches": list(arm.get("patches") or []),
            "exclusive_with": list(arm.get("exclusive_with") or []),
            "other_metrics": list(arm.get("other_metrics") or []),
            "primary_metric_note": _s(primary.get("note")),
            "seeded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    return {"cid": cid, "kwargs": kwargs}


def merge(a: dict, b: dict) -> dict:
    """Merge two rows that collapsed onto the same canonical_id.

    The row with the higher best_throughput owns best_config; knowledge lists
    concatenate (dedup by their primary text key); sessions append.
    """
    ka, kb = a["kwargs"], b["kwargs"]
    winner, loser = (ka, kb) if ka["best_throughput"] >= kb["best_throughput"] else (kb, ka)
    for field, key in (
        ("what_worked", "description"),
        ("what_failed", "description"),
        ("pitfalls", "description"),
        ("remaining_gaps", "description"),
        ("lessons", "statement"),
    ):
        seen = {_s(item.get(key)) for item in winner[field]}
        winner[field] += [item for item in loser[field] if _s(item.get(key)) not in seen]
    winner["sessions"] = list(winner["sessions"]) + list(loser["sessions"])
    winner["evidence_refs"] = sorted(set(winner["evidence_refs"]) | set(loser["evidence_refs"]))
    both = [
        winner["provenance"].get("recipe_id", ""),
        loser["provenance"].get("recipe_id", ""),
    ]
    winner["provenance"]["merged_from"] = [x for x in dict.fromkeys(both) if x]
    winner["extras"]["recipe_id"] = "+".join(x for x in both if x)
    if loser["confidence"] > winner["confidence"]:
        winner["confidence"] = loser["confidence"]
    return {"cid": a["cid"], "kwargs": winner}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="", help="KB root (default: $KNOWLEDGE_LOCAL_ROOT or <repo>/hyperloom/kb)")
    ap.add_argument("--extract-dir", default=str(EXTRACT_DIR), help="dir holding serving_*.json extractions")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    extract_dir = Path(args.extract_dir)

    if args.root:
        root = Path(args.root)
    else:
        import os

        env_root = _s(os.environ.get("KNOWLEDGE_LOCAL_ROOT"))
        root = Path(env_root) if env_root else REPO / "hyperloom" / "kb"

    arms: list[dict] = []
    for name in SERVING_FILES:
        path = extract_dir / name
        if not path.is_file():
            print(f"MISSING extraction: {path}", file=sys.stderr)
            return 2
        arms.extend(json.loads(path.read_text(encoding="utf-8")))
    print(f"loaded {len(arms)} serving arms from {extract_dir.name}/")

    rows: dict[str, dict] = {}
    for arm in arms:
        row = row_from_arm(arm)
        cid = row["cid"]
        if cid in rows:
            print(f"cid collision {cid}: merging {row['kwargs']['provenance']['recipe_id']}")
            rows[cid] = merge(rows[cid], row)
        else:
            rows[cid] = row

    if args.dry_run:
        for cid in sorted(rows):
            print(f"DRY  {cid}")
        return 0

    store = LocalRecipeStore(root=root)
    for cid in sorted(rows):
        res = store.put_recipe(**rows[cid]["kwargs"])
        c = res["counts"]
        print(
            f"put  v{res['version']} {'NEW' if res['created'] else 'UPD'}  {cid}"
            f"  (worked={c['what_worked']} failed={c['what_failed']} pitfalls={c['pitfalls']}"
            f" lessons={c['lessons']} gaps={c['remaining_gaps']})"
        )
    manifest = {
        "seeded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "/home/qiba/ai/docs/recipes (serving arms)",
        "hardware": HARDWARE,
        "rows": len(rows),
        "canonical_ids": sorted(rows),
    }
    (root / "seed_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"KB root: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
