#!/usr/bin/env python3
"""Build the mi250x-recipe-ops skill data files from KB extractions.

Input : .tmp/kb_extract/{serving_A,serving_B,serving_C,knobs_ops,envs_patches}.json
Output: local-skills/mi250x-recipe-ops/data/{arms,knobs,ops,environments,patches}.json
        + data/recipe_kb.json (canonical id index over the seeded KB root)

Regenerate whenever a recipe changes (after re-running the extraction).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXTRACT = REPO / ".tmp" / "kb_extract"
DATA = REPO / "local-skills" / "mi250x-recipe-ops" / "data"


def load(name: str) -> list[dict]:
    path = EXTRACT / name
    if not path.is_file():
        print(f"missing extraction: {path}", file=sys.stderr)
        raise SystemExit(2)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    arms = load("serving_A.json") + load("serving_B.json") + load("serving_C.json")
    knobs_ops = load("knobs_ops.json")
    envs_patches = load("envs_patches.json")

    DATA.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def dump(name: str, payload: dict) -> None:
        (DATA / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote data/{name}")

    dump(
        "arms.json",
        {
            "generated_at": stamp,
            "source": "/home/qiba/ai/docs/recipes/serving",
            "hardware": "8x MI250X gfx90a, ROCm 7.2.4, power cap 560W/module",
            "arms": sorted(arms, key=lambda a: str(a.get("port", ""))),
        },
    )
    dump(
        "knobs.json",
        {
            "generated_at": stamp,
            "source": "/home/qiba/ai/docs/recipes/knobs",
            "knobs": [x for x in knobs_ops if x.get("kind") == "knob"],
        },
    )
    dump(
        "ops.json",
        {
            "generated_at": stamp,
            "source": "/home/qiba/ai/docs/recipes/ops",
            "ops": [x for x in knobs_ops if x.get("kind") == "op"],
        },
    )
    dump(
        "environments.json",
        {
            "generated_at": stamp,
            "source": "/home/qiba/ai/docs/recipes/environments",
            "environments": [x for x in envs_patches if x.get("kind") == "environment"],
        },
    )
    dump(
        "patches.json",
        {
            "generated_at": stamp,
            "source": "/home/qiba/ai/docs/recipes/patches",
            "patches": [x for x in envs_patches if x.get("kind") == "patch"],
        },
    )

    kb_root = REPO / "hyperloom" / "kb"
    row_paths = sorted(kb_root.rglob("recipe.json")) if kb_root.is_dir() else []
    rows = sorted(str(p.parent.relative_to(kb_root)) for p in row_paths)

    def digest(path: Path) -> dict:
        """Compact per-row index entry (no bulk knowledge text)."""
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # unreadable row: index it, do not fail the build
            return {"dir": str(path.parent.relative_to(kb_root)), "error": str(exc)}
        entry = {
            "dir": str(path.parent.relative_to(kb_root)),
            "canonical_id": r.get("canonical_id"),
            "model": r.get("model"),
            "hardware": r.get("hardware"),
            "precision": r.get("precision"),
            "framework_version": r.get("framework_version"),
            "status": r.get("status"),
            "authority": r.get("authority"),
            "confidence": r.get("confidence"),
            "best_throughput": r.get("best_throughput"),
            "counts": {
                k: len(r.get(k) or [])
                for k in ("lessons", "pitfalls", "what_worked", "what_failed", "remaining_gaps")
            },
        }
        if r.get("vendor_hardware_matrix"):
            entry["vendor_support_gate"] = {
                "verified": sorted((r.get("vendor_hardware_matrix") or {}).get("verified", {})),
                "on_this_hardware": (r.get("mi250x_applicability") or {}).get("verdict"),
            }
        return entry

    digests = [digest(p) for p in row_paths]
    manifest_path = kb_root / "seed_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    dump(
        "recipe_kb.json",
        {
            "generated_at": stamp,
            "kb_root": str(kb_root),
            "env_hint": f"KNOWLEDGE_LOCAL_ROOT={kb_root}",
            "seed_manifest": manifest,
            "recipe_dirs": rows,
            "rows": digests,
        },
    )

    ref_path = EXTRACT / "qwen3_5_397b_reference.json"
    if ref_path.is_file():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        dump(
            "vendor_references.json",
            {
                "generated_at": stamp,
                "note": (
                    "Vendor (vllm-project/recipes) recipes folded in as *reference*, not as "
                    "MI250X-validated configs. Each entry separates the hardware-independent "
                    "flags that transfer to gfx90a from what the vendor's own support matrix "
                    "gates to MI300X+."
                ),
                "references": [
                    {
                        "hf_id": ref["identity"]["hf_id"],
                        "page": ref["source"]["page"],
                        "json_api": ref["source"]["json_api"],
                        "date_updated": ref["source"]["date_updated"],
                        "architecture": ref["identity"]["architecture"],
                        "model_type": ref["identity"]["vllm_model_type"],
                        "architectures": ref["identity"]["vllm_architectures"],
                        "vendor_hardware_matrix": ref["vendor_hardware_matrix"],
                        "base_args": ref["base_args"],
                        "base_env": ref["base_env"],
                        "mi250x_applicability": ref["mi250x_applicability"],
                        "troubleshooting": ref["troubleshooting"],
                        "vendor_commands": ref["vendor_commands"],
                        "artifact": f".tmp/kb_extract/{ref_path.name}",
                    }
                ],
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
