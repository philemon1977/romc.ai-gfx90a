#!/usr/bin/env python3
"""Summarize the single-stream lab: official InferenceX result JSONs -> one table.

Reads .tmp/single_stream_lab/<config>/*.json (each written by
utils/bench_serving/benchmark_serving.py through benchmarks/vllm_mi250x.sh) plus
the agentic prefix_probe.json files, and prints per-point decode speed, TTFT and
TPOT with the A/B/C comparison. Numbers are official-client semantics:
CONC=1 means one stream, --ignore-eos, random dataset, --num-warmups 2*CONC.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
CONFIGS = ["A_nospec", "B_mtp", "C_256k"]


def rows(cfg: Path) -> list[dict]:
    out = []
    for f in sorted(cfg.glob(f"{cfg.name}_c*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception as exc:  # noqa: BLE001 - a truncated json is a row to report, not a crash
            out.append({"file": f.name, "error": str(exc)})
            continue
        comp = d.get("completed") or 0
        dur = d.get("duration") or 0.0
        tok = d.get("total_output_tokens") or 0
        per_stream = (tok / dur) if (dur and comp) else None
        out.append(
            {
                "file": f.name,
                "conc": d.get("max_concurrency"),
                "isl": d.get("input_len") or (d.get("input_lens") or [None])[0],
                "osl_target": (d.get("output_lens") or [None])[0],
                "completed": comp,
                "duration_s": round(dur, 1),
                "out_tok": tok,
                "output_throughput": d.get("output_throughput"),
                "per_stream_tps": round(per_stream, 2) if per_stream else None,
                "ttft_mean_ms": d.get("mean_ttft_ms"),
                "ttft_med_ms": d.get("median_ttft_ms"),
                "ttft_p99_ms": d.get("p99_ttft_ms"),
                "tpot_mean_ms": d.get("mean_tpot_ms"),
                "tpot_med_ms": d.get("median_tpot_ms"),
                "itl_med_ms": d.get("median_itl_ms"),
                "error": d.get("error"),
            }
        )
    return out


def key(r: dict) -> tuple:
    return (r.get("conc") or 0, -(r.get("isl") or 0))


def main() -> int:
    data = {}
    for name in CONFIGS:
        cfg = LAB / name
        if cfg.is_dir():
            data[name] = [r for r in rows(cfg)]
    if not data:
        print("no result dirs yet")
        return 1

    for name, rs in data.items():
        print(f"\n=== {name} ===")
        print(f"{'conc':>4} {'isl':>7} {'done':>5} {'s/it':>6} {'out_tps':>8} {'per_stream':>10} "
              f"{'ttft_med':>9} {'tpot_med':>8} {'itl_med':>8}")
        for r in sorted(rs, key=key):
            if r.get("error"):
                print(f"  {r['file']}: ERROR {str(r['error'])[:80]}")
                continue
            st = (r["duration_s"] / r["completed"]) if r.get("completed") else None
            print(f"{r['conc']:>4} {r['isl']:>7} {r['completed']:>5} {st or 0:>6.1f} "
                  f"{r['output_throughput'] or 0:>8.2f} {r['per_stream_tps'] or 0:>10.2f} "
                  f"{r['ttft_med_ms'] or 0:>9.1f} {r['tpot_med_ms'] or 0:>8.2f} {r['itl_med_ms'] or 0:>8.2f}")

    # MTP speedup at matched points, single stream only
    if "A_nospec" in data and "B_mtp" in data:
        b = {(r["conc"], r["isl"]): r for r in data["B_mtp"] if not r.get("error")}
        print("\n=== B_mtp vs A_nospec (single stream) ===")
        for r in data["A_nospec"]:
            if r.get("error") or r["conc"] != 1:
                continue
            m = b.get((1, r["isl"]))
            if not m or not m.get("per_stream_tps") or not r.get("per_stream_tps"):
                continue
            sp = m["per_stream_tps"] / r["per_stream_tps"]
            tt = (m["ttft_med_ms"] / r["ttft_med_ms"]) if (m.get("ttft_med_ms") and r.get("ttft_med_ms")) else None
            print(f"  isl={r['isl']:>6}  decode {r['per_stream_tps']:>6.2f} -> {m['per_stream_tps']:>6.2f} tok/s "
                  f"({sp:.2f}x)  ttft {r['ttft_med_ms']:>8.1f} -> {m['ttft_med_ms']:>8.1f} ms "
                  f"({tt:.2f}x)" if tt else f"  isl={r['isl']:>6}  {sp:.2f}x")

    for name in data:
        p = LAB / name / "prefix_probe.json"
        if p.is_file():
            d = json.loads(p.read_text())
            print(f"\n=== {name} agentic turns (shared prefix) ===")
            print(f"  context {d['context_tokens_measured']} tok | cold TTFT {d['cold_ttft_ms']} ms | "
                  f"warm TTFT median {d['warm_ttft_ms_median']} ms | prefix-cache speedup {d['prefix_cache_speedup']}x | "
                  f"decode median {d['decode_tps_median']} tok/s")
            for t in d.get("turns", []):
                print(f"    {t['tag']}: prompt {t['prompt_tokens']} tok, TTFT {t['ttft_ms']} ms, "
                      f"decode {t['decode_tps']} tok/s, e2e {t['e2e_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
