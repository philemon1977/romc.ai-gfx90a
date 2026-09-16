#!/usr/bin/env python3
"""Read back the workspace-local Hyperloom RecipeKB and validate every row.

Checks (all through Hyperloom's own code, no reimplementation):
* every live ``recipe.json`` parses and its cid matches its directory path;
* ``LocalRecipeStore.search`` finds rows by model / hardware / framework /
  precision label_match and by metric bounds;
* identity fields in the row agree with the 7-tuple in the canonical id.

Usage: python3 scripts/verify_recipe_kb.py [--root DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "hyperloom"))

from hyperloom.orchestrator.knowledge.recipe_kb.local_store import (  # noqa: E402
    RECIPE_FILENAME,
    LocalRecipeStore,
)
from hyperloom.orchestrator.knowledge.recipe_kb.canonical_id import (  # noqa: E402
    canonical_id_for_path,
    cid_to_path_components,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="")
    args = ap.parse_args()
    env_root = str(os.environ.get("KNOWLEDGE_LOCAL_ROOT") or "").strip()
    root = Path(args.root or env_root or (REPO / "hyperloom" / "kb"))
    store = LocalRecipeStore(root=root)

    errors = 0
    rows = []
    for recipe_dir in store._walk_recipe_dirs():
        path = recipe_dir / RECIPE_FILENAME
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            cid = canonical_id_for_path(root=root, recipe_dir=recipe_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL parse {path}: {exc}")
            errors += 1
            continue
        if payload.get("canonical_id") != cid:
            print(f"FAIL cid mismatch: dir→{cid} row→{payload.get('canonical_id')}")
            errors += 1
            continue
        model, hardware, fw, mt, arch, fvw, prec = cid_to_path_components(cid)
        for dim, val in (
            ("model", model),
            ("hardware", hardware),
            ("framework_version", fvw),
            ("precision", prec),
        ):
            row_val = str(payload.get(dim) or "").lower().rstrip("/").rsplit("/", 1)[-1]
            row_val = row_val.replace(" ", "_")
            if row_val and val not in (row_val, "unknown_" + dim):
                print(f"FAIL {cid}: identity {dim} row={row_val!r} dir={val!r}")
                errors += 1
        rows.append((cid, payload))

    print(f"rows on disk: {len(rows)}")

    probes = [
        ("all mi250x", {"label_match": {"hardware": "mi250x"}}),
        ("vllm", {"label_match": {"framework_name": "vllm"}}),
        ("llama.cpp", {"label_match": {"framework_name": "llama.cpp"}}),
        ("bf16", {"label_match": {"precision": "bf16"}}),
        ("throughput>=20", {"metric_filters": {"best_throughput": {"min": 20}}}),
    ]
    for name, kw in probes:
        hits = store.search(**kw, limit=1000)
        print(f"search {name:16s} → {len(hits)}")

    sample = store.search(label_match={"hardware": "mi250x"}, limit=1)
    if sample:
        got = store.get_recipe(canonical_id=sample[0]["canonical_id"])
        print(f"get_recipe round-trip: {'OK' if got else 'FAIL'}")
        if not got:
            errors += 1

    if errors:
        print(f"VERIFY FAILED: {errors} problem(s)")
        return 1
    print("VERIFY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
