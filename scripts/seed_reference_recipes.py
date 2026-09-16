#!/usr/bin/env python3
"""Fold the official vLLM recipe for ``Qwen/Qwen3.5-397B-A17B`` — the base model
of ``Ornith-1.5-397B`` — into the workspace RecipeKB, translated to what is
actually decidable on MI250X (gfx90a).

Source artifact: ``.tmp/kb_extract/qwen3_5_397b_reference.json`` (harvested from
https://recipes.vllm.ai/Qwen/Qwen3.5-397B-A17B.json, 2026-09-15).

Why a translation is needed at all: the vendor matrix for this model is
``{h200, gb200, mi300x, mi325x, mi355x, ascend_950dt}`` — MI250X is absent, and
the whole official AMD lane runs the FP8 checkpoint with FP8 GEMM. gfx90a has no
FP8 matrix core and ``torch._scaled_mm`` refuses anything below MI300+, so the
recipe's hardware-dependent half is a silicon gate, not a config choice. Only
its hardware-independent half (``--language-model-only``, DeepGEMM off,
``--trust-remote-code``, the hybrid Mamba/GDN cache pitfalls, MoE parallelism
shape) transfers.

Three writes, idempotent (list-valued knowledge merges on a stable key):
  1. reference row for the parent, FP8 lane   -> what the vendor ships + the gate
  2. reference row for the parent, BF16 lane  -> the capacity gate (780 GiB > HBM)
  3. merge reference-derived lessons/pitfalls into the live Ornith arm row

Usage:
    PYTHONPATH=hyperloom python3 scripts/seed_reference_recipes.py [--root DIR] [--dry-run]
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

HARDWARE = "mi250x"
FRAMEWORK_VERSION = "0.28.0"
MODEL_TYPE = "qwen3_5_moe"
ARCH = "qwen3_5moeforconditionalgeneration"
REFERENCE_ARTIFACT = ".tmp/kb_extract/qwen3_5_397b_reference.json"

ORNITH_CID = (
    f"inference:ornith-1.5-397b-fp8:{HARDWARE}:vllm:{MODEL_TYPE}:{ARCH}:"
    f"{FRAMEWORK_VERSION}:fp8"
)


def cid(slug: str, precision: str) -> str:
    return f"inference:{slug}:{HARDWARE}:vllm:{MODEL_TYPE}:{ARCH}:{FRAMEWORK_VERSION}:{precision}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _key(entry: dict) -> str:
    """Stable merge key for a knowledge entry (description/statement prefix)."""
    for field in ("description", "statement"):
        if entry.get(field):
            return str(entry[field])[:80]
    return json.dumps(entry, sort_keys=True, ensure_ascii=False)[:80]


def merge(existing: list | None, additions: list[dict]) -> tuple[list, int]:
    """Append additions that are not already present (keyed by prefix)."""
    out = list(existing or [])
    seen = {_key(e) for e in out if isinstance(e, dict)}
    added = 0
    for entry in additions:
        if _key(entry) in seen:
            continue
        out.append(entry)
        seen.add(_key(entry))
        added += 1
    return out, added


_CMP_FIELDS = (
    "best_config", "best_throughput", "what_worked", "what_failed",
    "remaining_gaps", "pitfalls", "lessons", "authority", "confidence",
    "evidence_refs",
)


def _unchanged(payload: dict, row: dict | None) -> bool:
    """True when a reference row already carries exactly this payload.

    ``put_recipe`` bumps ``version`` (and archives history) on every call, so a
    re-run of this seeder must not write when nothing changed. On-disk rows
    splat ``extras`` at the top level, so those keys are compared individually.
    """
    if row is None:
        return False
    for field in _CMP_FIELDS:
        if _canon(payload.get(field)) != _canon(row.get(field)):
            return False
    for key, value in (payload.get("extras") or {}).items():
        if key == "seeded_at":
            continue
        if _canon(value) != _canon(row.get(key)):
            return False
    return True


def _canon(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def load_artifact() -> dict:
    path = REPO / REFERENCE_ARTIFACT
    if not path.is_file():
        raise SystemExit(f"missing reference artifact: {path}")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# 1+2. parent reference rows
# --------------------------------------------------------------------------- #

def reference_kwargs(art: dict, precision: str) -> dict:
    src = art["source"]
    ident = art["identity"]
    matrix = art["vendor_hardware_matrix"]
    app = art["mi250x_applicability"]
    # Slug must equal what the optimizer derives from the checkpoint directory
    # name, so a real run on that directory matches this row exactly.
    vendor_slug = "qwen3.5-397b-a17b-fp8" if precision == "fp8" else "qwen3.5-397b-a17b"

    common_lessons = [
        {
            "statement": (
                "官方 vLLM 配方 (vllm-project/recipes, models/Qwen/Qwen3.5-397B-A17B.yaml) 对 AMD "
                "只验证到 MI300X/MI325X/MI355X；MI250X 不在支持矩阵内，官方 AMD 轮子明确写 "
                "\"Supported GPUs: MI300X, MI325X, MI355X\""
            ),
            "measured_impact": "厂商矩阵 {h200, gb200, mi300x, mi325x, mi355x, ascend_950dt}，无 mi250x / mi350x",
        },
        {
            "statement": (
                "该配方 AMD 侧整条线都跑 FP8 checkpoint + FP8 GEMM（Qwen/Qwen3.5-397B-A17B-FP8，"
                "W8A8 动态 per-token 激活）。gfx90a 无 FP8 矩阵核，torch._scaled_mm 硬门 >=MI300+，"
                "所以这是硅层面的门，不是可调参数"
            ),
            "measured_impact": "本机 fp8 首次 linear apply 即 die，错误文本与配方硬件前提一致",
        },
        {
            "statement": (
                "同一身份 (model_type=qwen3_5_moe / architectures=[Qwen3_5MoeForConditionalGeneration])，"
                "因此配方的架构级旗标可用：--language-model-only、VLLM_USE_DEEP_GEMM=0、--trust-remote-code，"
                "以及 hybrid GDN+Mamba 状态缓存的坑"
            ),
            "measured_impact": "Ornith-1.5-397B-FP8 与本模型同 model_type/architectures，故可对标",
        },
    ]

    common_pitfalls = [
        {
            "description": (
                "hybrid GDN+Mamba 状态缓存会限死 CUDA graph 捕获：vendor 配方本模型的故障项 "
                "\"assert num_cache_lines >= batch\"（graph capture size > Mamba state cache），"
                "修法是调低 --max-cudagraph-capture-size（默认 512）。我们自己的启动日志已经出现同类形态："
                "\"Setting attention block size to 528 tokens to ensure attention page size >= mamba page size\" "
                "+ \"Padding mamba page size by 1.34%\""
            ),
            "severity": "crash",
        },
        {
            "description": (
                "MTP：checkpoint 自带 mtp.* 权重（1553 个张量），故 --speculative-config mtp 是模型级可用杠杆；"
                "但 vendor 明确标注 AMD 上 MTP-1 speculative decoding 仍在开发中，且 MTP 在高并发下反而吃 KV/降吞吐。"
                "要用必须先看 acceptance，再判收益"
            ),
            "severity": "regress",
        },
    ]

    common_gaps = [
        {
            "description": (
                "MI250X 上该身份的活着配置尚未测出：BF16 权重 780 GiB > 512 GiB HBM（不可行），"
                "FP8 需要 gfx90a 反量化仿真内核（本地补丁已能加载，未见健康端点/端到端吞吐）"
            ),
            "metrics": "N/A",
        }
    ]

    feats = art.get("vendor_commands") or {}
    if precision == "fp8":
        best_config = {
            "extra_server_args": (
                "--trust-remote-code --language-model-only --reasoning-parser qwen3 "
                "--tensor-parallel-size 8"
            ),
            "extra_envs": {"VLLM_USE_DEEP_GEMM": "0", "VLLM_DEEP_GEMM_WARMUP": "skip"},
        }
        what_failed = [
            {
                "description": (
                    "在 MI250X 上照抄官方 AMD 命令（Qwen/Qwen3.5-397B-A17B-FP8 + FP8 GEMM）"
                    "必然 server_init_dead"
                ),
                "reason": (
                    "vendor 的 AMD 前提是 MI300X+（FP8 MMA）。gfx90a 上报的 capability 被 Torch 认成 (9,0)，"
                    "反而绕过 cc 门，最终在 scaled_mm/pytorch.py:227 抛出 "
                    "'only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+'"
                ),
            }
        ]
        what_worked = [
            {
                "description": (
                    "同身份的本地反量化仿真路线已能过权重加载：TP8 下 'Model loading took 48.27 GiB memory "
                    "and 266 s'，随后完成 hybrid cache 分页对齐并进入 torch.compile"
                ),
                "measured_impact": "weights load OK；健康端点与端到端吞吐未测（session 20260915T161838Z-ba2694ea）",
            }
        ]
        extras = {
            "status": "reference",
            "vendor_recipe": {
                "page": src["page"],
                "json_api": src["json_api"],
                "date_updated": src["date_updated"],
                "variant": "fp8",
                "vendor_model_id": "Qwen/Qwen3.5-397B-A17B-FP8",
                "vendor_commands": {"text_only_throughput": feats.get("throughput_text_only"),
                                    "mi355x": feats.get("mi355x")},
            },
            "vendor_hardware_matrix": matrix,
            "mi250x_applicability": app,
            "reference_artifact": REFERENCE_ARTIFACT,
        }
    else:  # bf16
        best_config = {
            "extra_server_args": "--trust-remote-code --language-model-only --tensor-parallel-size 8",
            "extra_envs": {"VLLM_USE_DEEP_GEMM": "0", "VLLM_DEEP_GEMM_WARMUP": "skip"},
        }
        what_failed = [
            {
                "description": "MI250X 上 BF16 权重不可行（容量门，不是内核门）",
                "reason": (
                    "397B BF16 = 780 GiB > 本机 8x64 GiB = 512 GiB HBM；官方 BF16 变体本身就要 8x H200"
                    "（vram_minimum_gb 953）"
                ),
            }
        ]
        what_worked = []
        extras = {
            "status": "reference",
            "vendor_recipe": {
                "page": src["page"],
                "json_api": src["json_api"],
                "date_updated": src["date_updated"],
                "variant": "bf16",
                "vendor_model_id": ident["hf_id"],
            },
            "vendor_hardware_matrix": matrix,
            "mi250x_applicability": app,
            "reference_artifact": REFERENCE_ARTIFACT,
        }

    return dict(
        canonical_id=cid(vendor_slug, precision),
        model="Qwen3.5-397B-A17B" + ("-FP8" if precision == "fp8" else ""),
        hardware=HARDWARE,
        framework_name="vllm",
        framework_version=FRAMEWORK_VERSION,
        precision=precision,
        best_config=best_config,
        best_throughput=0.0,
        what_worked=what_worked,
        what_failed=what_failed,
        remaining_gaps=common_gaps,
        pitfalls=common_pitfalls,
        lessons=common_lessons,
        last_profiled="2026-09-15",
        stack_fingerprint={"vllm_version": "0.28.0+rocm723", "rocm_version": "7.2.4"},
        authority="VENDOR_REFERENCE",
        confidence=0.95,
        evidence_refs=[src["page"], src["json_api"], str(REPO / REFERENCE_ARTIFACT)],
        extras={
            **extras,
            "architecture": ident["architecture"],
            "parameter_count": ident["total_params"],
            "active_parameters": ident["active_params"],
            "context_length": ident["context_length"],
            "min_vllm_version": ident["min_vllm_version"],
            "base_args": art["base_args"],
            "base_env": art["base_env"],
            "gpus": "mi250dx8",
            "rocm_version": "7.2.4",
            "tp": 8,
            "ep": 1,
            "seeded_at": _now(),
        },
    )


# --------------------------------------------------------------------------- #
# 3. merge into the live Ornith arm row
# --------------------------------------------------------------------------- #

def ornith_reference_knowledge(art: dict) -> dict:
    src = art["source"]
    feats = art.get("vendor_commands") or {}
    lessons = [
        {
            "statement": (
                "基座 Qwen3.5-397B-A17B 的官方 vLLM 配方 AMD 侧最低只到 MI300X/MI325X/MI355X，"
                "整条官方 AMD 线是 FP8 checkpoint+FP8 GEMM。所以本臂的 FP8 死在 gfx90a 是硅层面的门"
                "（无 FP8 矩阵核 / scaled_mm 硬门），不是参数没调对——继续在配置层试 fp8 没有意义，"
                "要么走反量化仿真内核，要么换权重精度"
            ),
            "measured_impact": "官方矩阵 {h200, gb200, mi300x, mi325x, mi355x, ascend_950dt}，无 mi250x",
        },
        {
            "statement": (
                "官方配方里对本臂真正可搬的是硬件无关项：--language-model-only（text-only 省下 vision tower 的显存）、"
                "VLLM_USE_DEEP_GEMM=0 + VLLM_DEEP_GEMM_WARMUP=skip（DeepGEMM 是 CUDA 专属，别让它 warmup）、"
                "--trust-remote-code、--enable-prefix-caching，以及 MoE 的 EP/-dp 形状"
            ),
            "measured_impact": "本臂权重 ~48-55 GiB/die（共 64 GiB），language-model-only 是唯一立刻见效的显存项",
        },
        {
            "statement": (
                "checkpoint 自带 mtp.* 权重（1553 张量）→ --speculative-config '{\"method\":\"mtp\","
                "\"num_speculative_tokens\":1}' 是模型级杠杆；但 vendor 标注 AMD 上 MTP 仍在开发，"
                "且 MTP 在并发下用 KV 换 TPOT。先测 acceptance 再谈收益"
            ),
            "measured_impact": "未测；vendor 明确 \"MTP-1 speculative decoding for AMD GPUs is under development\"",
        },
        {
            "statement": (
                "kernel 选择类补丁落进同一棵树时，vLLM 会重放按哈希缓存的 AOT graph——只改选择逻辑不换 cache root 不会生效；"
                "换 VLLM_CACHE_ROOT / TORCHINDUCTOR_CACHE_DIR 才让它重编（这是 161838Z 会话死基线的第二个叠加原因）"
            ),
            "measured_impact": "specialist f008270e 把两个原因并列：选择门 + AOT graph 重放",
        },
    ]
    pitfalls = [
        {
            "description": (
                "hybrid GDN+Mamba 状态缓存限死 CUDA graph 捕获：官方配方本模型的故障项 "
                "\"assert num_cache_lines >= batch\"，修法 --max-cudagraph-capture-size 调低（默认 512）。"
                "本臂启动日志已见同族形态：attention block size 528 / mamba page size padding 1.34% "
                "(interface.py:911/935)"
            ),
            "severity": "crash",
        },
        {
            "description": (
                "官方建议 text-only 时不要同时开 vision 相关项：--language-model-only 与 "
                "--mm-encoder-tp-mode data 互斥；encoder DP 会额外吃显存，需要同时调 --gpu-memory-utilization"
            ),
            "severity": "regress",
        },
    ]
    what_worked = [
        {
            "description": (
                "反量化仿真内核把 FP8 权重跑通了加载阶段（TP8）：'Model loading took 48.27 GiB memory and 266 s'，"
                "dense linear 在 load 时 FP8→BF16 反量化（emulation.py:175），随后 hybrid cache 分页对齐、进入 torch.compile"
            ),
            "measured_impact": "权重可加载；健康端点 / 端到端吞吐仍未测（session 20260915T161838Z-ba2694ea）",
        }
    ]
    gaps = [
        {
            "description": (
                "反量化仿真路线的端到端基线未拿下：加载通过但 /v1/models 未 ready，首次 torch.compile 很慢"
                "（Dynamo 12.4 s 起步 + shm broadcast 60 s 无块告警属编译期正常形态）"
            ),
            "metrics": "N/A",
        },
        {
            "description": (
                "官方硬件无关旗标里未在本臂试过的：--language-model-only、VLLM_USE_DEEP_GEMM=0、"
                "--enable-expert-parallel/-dp 形状、--kv-cache-dtype fp8（hybrid 缓存下的实际收益未知）"
            ),
            "metrics": "N/A",
        },
    ]
    return {
        "lessons": lessons,
        "pitfalls": pitfalls,
        "what_worked": what_worked,
        "remaining_gaps": gaps,
        "extras": {
            "status": "experimental",
            "status_note": (
                "原判 dead 针对\"无补丁的 stock vLLM fp8 路线\"；反量化仿真内核已证明可加载权重，"
                "端到端未测，故降级为 experimental（本行随后由 optimizer CLOSE 的 sediment 覆写）"
            ),
            "vendor_reference": {
                "hf_id": art["identity"]["hf_id"],
                "page": src["page"],
                "json_api": src["json_api"],
                "date_updated": src["date_updated"],
                "artifact": REFERENCE_ARTIFACT,
                "third_party_commands": {
                    "mi355x": feats.get("mi355x"),
                    "text_only_throughput": feats.get("throughput_text_only"),
                },
                "silicon_gated_on_gfx90a": art["mi250x_applicability"]["silicon_gated_on_gfx90a"],
                "transferable_to_mi250x": art["mi250x_applicability"]["transferable_to_mi250x"],
            },
        },
        "evidence_refs": [src["page"], str(REPO / REFERENCE_ARTIFACT)],
    }


def merge_ornith(store: LocalRecipeStore, art: dict, dry_run: bool) -> dict:
    live_path = store._live_path(ORNITH_CID)  # noqa: SLF001 - read-only inspection
    if not live_path.is_file():
        raise SystemExit(f"Ornith arm row not found: {live_path}")
    row = json.loads(live_path.read_text())
    add = ornith_reference_knowledge(art)

    kwargs = {
        "canonical_id": ORNITH_CID,
        "model": row.get("model", "Ornith-1.5-397B-FP8"),
        "hardware": row.get("hardware", HARDWARE),
        "framework_name": row.get("framework_name") or row.get("framework") or "vllm",
        "framework_version": row.get("framework_version", FRAMEWORK_VERSION),
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
    for field in ("what_worked", "what_failed", "remaining_gaps", "pitfalls", "lessons"):
        merged, added = merge(row.get(field), add.get(field, []))
        kwargs[field] = merged
        stats[field] = added
    refs = list(row.get("evidence_refs") or [])
    for ref in add["evidence_refs"]:
        if ref not in refs:
            refs.append(ref)
    kwargs["evidence_refs"] = refs

    extras = {k: v for k, v in row.items() if k not in {
        "canonical_id", "version", "created_at", "updated_at", "model", "hardware",
        "framework_name", "framework", "framework_version", "precision", "best_config",
        "best_throughput", "what_worked", "what_failed", "remaining_gaps", "pitfalls",
        "lessons", "last_profiled", "stack_fingerprint", "sessions", "authority",
        "confidence", "evidence_refs", "provenance", "_field_sources", "_sources",
        "prs_tested", "kernel_optimizations",
    }}
    if row.get("kernel_optimizations"):
        extras["kernel_optimizations"] = row["kernel_optimizations"]
    extras.update(add["extras"])
    extras["reference_folded_at"] = _now()
    kwargs["extras"] = extras

    nothing_new = (
        not any(stats.values())
        and all(_canon(row.get(k)) == _canon(v) for k, v in add["extras"].items())
        and all(ref in (row.get("evidence_refs") or []) for ref in add["evidence_refs"])
    )
    if nothing_new:
        return {"canonical_id": ORNITH_CID, "unchanged": True,
                "version": row.get("version"), "added": stats}
    if dry_run:
        return {"canonical_id": ORNITH_CID, "dry_run": True, "added": stats,
                "prior_version": row.get("version")}
    result = store.put_recipe(**kwargs)
    return {**result, "added": stats}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="KB root (default: repo hyperloom/kb)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    art = load_artifact()
    root = Path(args.root) if args.root else REPO / "hyperloom" / "kb"
    store = LocalRecipeStore(root=root)
    print(f"KB root: {root}")

    for precision in ("fp8", "bf16"):
        payload = reference_kwargs(art, precision)
        live_path = store._live_path(payload["canonical_id"])  # noqa: SLF001
        prior = json.loads(live_path.read_text()) if live_path.is_file() else None
        if _unchanged(payload, prior):
            print(f"unchanged {payload['canonical_id']} (version={prior.get('version')})")
            continue
        if prior is not None and (prior.get("seeded_at")):
            payload["extras"]["seeded_at"] = prior["seeded_at"]
        if args.dry_run:
            print(f"[dry-run] would write {payload['canonical_id']}")
            continue
        res = store.put_recipe(**payload)
        print(f"wrote {res['canonical_id']} version={res['version']} created={res['created']}")

    result = merge_ornith(store, art, args.dry_run)
    tag = "[dry-run] " if args.dry_run else ""
    print(f"{tag}ornith merge: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
