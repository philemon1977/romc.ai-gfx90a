#!/usr/bin/env python3
"""把 GLM-5.3-CT-Int4-W4A16 在 MI250X 上的实测结论写回 Hyperloom 知识库。

背景（2026-09-21 审计）：KB 里 GLM-5.3 int4 的 mi250x 槽位是 Hyperloom t0_anchor
自己建的**空壳**（best_config={} / what_worked=[] / evidence_refs=[]，version=2），
本会话攒下的补丁与结论一条都不在里面；而且那两条 recipe.json 原是 root:root 0600，
hyperloom 的 local_store.search() 读它会直接抛 LocalRecipeStoreError
（scripts/verify_recipe_kb.py 因此失败）。本脚本负责后一半：用 Hyperloom 自己的
LocalRecipeStore.put_recipe 写回内容（保持 7 元组目录契约与 history/vN 归档语义）。

纯 CPU，不碰 GPU、不起服务。

用法：
    PYTHONPATH=hyperloom python3 scripts/note_glm53_int4_kb.py --dry-run
    PYTHONPATH=hyperloom python3 scripts/note_glm53_int4_kb.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "hyperloom"))

from hyperloom.orchestrator.knowledge.recipe_kb import LocalRecipeStore  # noqa: E402

CID = (
    "inference:glm-5.3-ct-int4-w4a16:mi250x:vllm:glm_moe_dsa:"
    "glmmoedsaforcausallm:0.3.1.dev85+gdee37d891:w4a16"
)
R = "hyperloom/reports/models/glm53-int4"

BEST_CONFIG = {
    "conc": 32,
    "isl": 1024,
    "osl": 1024,
    "max_model_len": 32768,
    "note": (
        "口径：8 GCD / TP8（=DCP8，同机同一组 rank）/ W4A16 / temp0。"
        "best_throughput=59.5 是并发 32 档聚合 tok/s（GEMV v3 后 59.1），"
        "roofline T_mem(mi250x) 同参数 = 859 tok/s ⇒ 6.9%。"
        "单流数字必须带 ctx：ctx≈800 时 10.2 tok/s（v3 后 9.60–10.71），"
        "ctx=8192 时约 4 tok/s。"
    ),
    "extra_envs": {
        "VLLM_ROCM_USE_AITER": "0",
        "VLLM_ROCM_USE_AITER_MOE": "0",
        "MI250_MOE_GEMV": "1",
        "MI250_MOE_GEMV_MODULE": "mi250_moe_gemv_gs",
        "DSV41_IDX_AITER_KERNEL": "1",
        "MI250_SPARSE_SPLITK": "0",
        "FASTSAFETENSORS_UNIFIED_MEM": "1",
        "MI250_FST_MAX_BATCH_MB": "1920",
    },
    "extra_server_args": "--tensor-parallel-size 8 --max-model-len 32768",
}

WHAT_WORKED = [
    {
        "description": (
            "MoE 专家 GEMV 自写 kernel v3（scale 提到循环外）：生产分片形状下 "
            "gemm1 8.7x / gemm2 5.0x"
        ),
        "measured_impact": (
            "端到端单流解码 6.38-6.81 -> 9.60-10.71 tok/s，并发 32 聚合 33.8 -> 59.1 tok/s，"
            "事实召回仍 6/6（" + R + "/moe-gemv-scale-hoist.md）"
        ),
    },
    {
        "description": (
            "int4 覆盖完整：所有 Linear 已是 W4A16，bf16 只剩 router / shared_experts / "
            "indexer.wk / norm / embed / lm_head / MTP eh_proj"
        ),
        "measured_impact": "402 GB 权重压到 52.92 GiB/rank（TP8）",
    },
    {
        "description": (
            "all-reduce 后端在 gfx90a 上选定 PYNCCL（tp:0 与 dcp:0 两个组都选它）"
        ),
        "measured_impact": "141 -> 255 us 每次集合通信（vLLM CudaCommunicator 对非 tp 组会禁用全部快路径）",
    },
    {
        "description": (
            "DSV41_IDX_AITER_KERNEL=1：indexer decode 从上游 torch 回退（launcher 自陈按行主序读 "
            "SHUFFLE 页缓存、本机结果不可信）换成自研 gfx90a 内核（已三方对拍）"
        ),
        "measured_impact": (
            "同一把尺子（请求级吞吐 ISL~700/OSL=128）：conc1 4.18->4.54 (+8.6%)、" 
            "conc8 26.64->32.51 (+22.0%)、conc32 73.13->123.78 (+69.3%)，召回 6/6；"
            "该 env 在当前 launcher 里默认未设 ⇒ 默认值是更慢且更不可信的那一支"
        ),
    },
]

WHAT_FAILED = [
    {
        "description": (
            "sparse-attention split-K（MI250_SPARSE_SPLITK=8）三档全负：kernel 级 7.2x "
            "（attention 861 -> 142 us）但 NCCL 141 -> 255 us、elementwise/copy 调用 1332 -> 3439/step；"
            "单杠杆消融（同一 indexer 内核下）conc1 -18.3% / conc8 -14.1% / conc32 -13.0%"
        ),
        "reason": "end_to_end_regression",
        "metrics": "MAXM 默认 8 时 conc32 根本不进场 ⇒ 高并发档必须把 MAXM 提到 32 才算测过",
    },
    {
        "description": (
            "QuickReduce C2+C3（三条 env：QUANTIZATION=FP / MIN_SIZE_BYTES_MB=0 / CAST_BF16_TO_FP16=0）"
            "在本模型上不可用：启用后 ops.init_custom_qr 固定占 ~9 GiB/卡，而本模型 KV 总预算仅 "
            "8.17 GiB ⇒ 启动显存门直接失败（free 54.9 < desired 62.06）；"
            "VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=16 压不动它（失败臂数字一模一样）"
        ),
        "reason": "memory_incompatible",
        "metrics": "4 个开 QR 的臂全部死在启动检查；不开 QR 的同刻 free 为 62.9-63.03 GiB，差值 ~9 GiB/卡"
    },
    {
        "description": "MTP 投机解码单流 -20%（接受长度 1.18-1.67），KV 池 326,016 -> 173,184",
        "reason": "end_to_end_regression",
    },
    {
        "description": "DCP 全不开（DCP=0）单流并不更快：7.66 vs DCP=8 的 9.60 tok/s",
        "reason": "hypothesis_falsified",
    },
    {
        "description": (
            "AITER MoE 路径在 gfx90a 不可用；FP8 路线不存在（CDNA2 无 FP8 矩阵核，"
            "首个 fp8 linear 就撞 torch._scaled_mm 引擎级门）"
        ),
        "reason": "unsupported_hardware",
    },
]

PITFALLS = [
    {
        "description": (
            "kernel 级收益不能外推到端到端：split-K 7.2x 换来 -8..-19%。"
            "同一形状下要同时看 attention / NCCL / elementwise 三类计数"
        ),
        "severity": "methodology",
    },
    {
        "description": (
            "报告单流 TPS 必须带上下文长度：ctx≈800 约 10 tok/s，ctx=8192 约 4 tok/s，"
            "不带 ctx 的数字没有意义"
        ),
        "severity": "methodology",
    },
    {
        "description": (
            "QuickReduce 在 CDNA2 上只有 FP 模式安全（与 NCCL 位一致）；"
            "CUSTOM(ca_comm) 在 gfx90a 会静默算错（输出全 !!!）——不要再探"
        ),
        "severity": "correctness",
    },
    {
        "description": (
            "HIP_VISIBLE_DEVICES 会泄进 ROCR 掩码子环境，import vllm 直接抛；"
            "起服前必须显式挑卡并先查显存（本机 8 GCD 常被其它会话占用）"
        ),
        "severity": "crash",
    },
    {
        "description": "MI250_SPARSE_SPLITK 的默认值必须是 0；开着会在生产档位静默变慢",
        "severity": "regression",
    },
]

LESSONS = [
    {
        "statement": (
            "GLM-5.3 int4 的解码是延迟/启动受限，不是带宽受限："
            "权重流量只有 18.6 GB/s/rank = 峰值的 1.1%"
        ),
        "measured_impact": (
            "ctx=8192/M=1/DCP=8 一步 186.7 ms GPU 工作量，其中 sparse-attn 67.2 (36%)、"
            "NCCL 36.3 (19.4%)、triton_w4a16_gemm 35.3 (18.9%)、我们的 MoE GEMV 3.3 (1.8%)"
        ),
    },
    {
        "statement": "层内串行开销吃掉 93% 的访存预算——这是“到不了 30 tok/s”最硬的表述",
        "measured_impact": "并发 32 实测 59.5 tok/s 对 roofline T_mem 859 tok/s = 6.9%",
    },
    {
        "statement": "一项修复、两把尺子不等于两笔成果：同一修复的两个 harness 数字不可相加",
        "measured_impact": "沿用本仓证据纪律",
    },
]

# 注意：Gap 是 dict（description/metrics），不是字符串——写字符串会被 schema 静默丢掉（2026-09-21 踩过）
REMAINING_GAPS = [
    {
        "description": "1M 上下文未达成：当前定稿档 32k，下一步先 256k 实测再谈 1M",
        "metrics": "KV 池 326,016 tokens（32k 档）；1M 未测",
    },
    {
        "description": "QuickReduce C2+C3 在本模型上未验证（三条 env 未启用，需起服 + 事实召回 6/6 + TPS A/B）",
        "metrics": "",
    },
    {
        "description": "DCP 的 all-gather 在 gfx90a 没有快后端 ⇒ 只能减少/融合 gather",
        "metrics": "AITER_CUSTOM 仅 MI300 提供 AG/RS；all_gather 不走快路径",
    },
    {
        "description": "DCP 补丁队列与挂载 tree 不一致：队列缺 _DCP_TOPK_CTX 与 split-K",
        "metrics": "见 " + R + "/patch-review-20260921.md",
    },
    {
        "description": "sparse split-K 的 7.2x 要变成净收益，需同时压掉它带来的 elementwise/copy 与 NCCL 增量",
        "metrics": "elementwise/copy 1332 -> 3439/step；NCCL 141 -> 255 us",
    },
]

# KernelOptimization 是定长 dataclass：键名/类型不对会被静默归零（2026-09-21 踩过）
_T = "2026-09-21"
KERNEL_OPTIMIZATIONS = [
    {
        "kernel_id": "mi250_moe_gemv_gs_v3",
        "source_file": "quark-int8/moe_gemv_patch/mi250_moe_gemv_gs.py",
        "artifact_path": "quark-int8/moe_gemv_patch/mi250_moe_gemv_gs.py",
        "micro_speedup": 8.7,
        "decision": "KEEP",
        "e2e_gain_pct": 74.85,
        "e2e_tput": 59.1,
        "integrated": True,
        "e2e_decision": "KEEP",
        "ts": _T,
    },
    {
        "kernel_id": "sparse_attn_splitk",
        "source_file": "quark-int8/dcp_patches/（tree 侧未入队列）",
        "artifact_path": "",
        "micro_speedup": 7.2,
        "decision": "REJECT",
        "e2e_gain_pct": -12.0,
        "e2e_tput": 8.65,
        "integrated": True,
        "e2e_decision": "REVERT",
        "ts": _T,
    },
    {
        "kernel_id": "dcp_partial_out_lse_merge",
        "source_file": "quark-int8/dcp_patches/0001-0006",
        "artifact_path": "",
        "micro_speedup": 1.0,
        "decision": "KEEP",
        "e2e_gain_pct": 0.0,
        "e2e_tput": 0.0,
        "integrated": True,
        "e2e_decision": "KEEP_CORRECTNESS_ONLY",
        "ts": _T,
    },
]

EVIDENCE_REFS = [
    R + "/moe-gemv-scale-hoist.md",
    R + "/decode-step-attribution.md",
    R + "/decode-splitk-attention.md",
    R + "/patch-review-20260921.md",
    R + "/hyperloom-run-20260921.md",
]


def row() -> dict:
    return dict(
        canonical_id=CID,
        model="GLM-5.3-CT-Int4-W4A16",
        hardware="mi250x",
        framework_name="vllm",
        framework_version="0.3.1.dev85+gdee37d891",
        precision="w4a16",
        best_config=BEST_CONFIG,
        best_throughput=59.5,
        what_worked=WHAT_WORKED,
        what_failed=WHAT_FAILED,
        pitfalls=PITFALLS,
        lessons=LESSONS,
        remaining_gaps=REMAINING_GAPS,
        last_profiled="2026-09-21",
        stack_fingerprint={
            "vllm_version": "0.3.1.dev85+gdee37d891.rocm723",
            "rocm_version": "7.2.3",
            "aiter_commit": "",
        },
        authority="EXPERIENTIAL",
        confidence=0.8,
        evidence_refs=EVIDENCE_REFS,
        provenance={
            "generator": "note_glm53_int4_kb",
            "source": "hyperloom/reports/models/glm53-int4 (本会话实测)",
            "filled_at": "2026-09-21",
            "note": (
                "槽位原为 Hyperloom t0_anchor 的空壳（sid 20260920T193320Z-cd513422），"
                "本脚本用 put_recipe 写回实测结论；口径与出处在 evidence_refs"
            ),
        },
        extras={
            "architectures": ["GlmMoeDsaForCausalLM"],
            "model_type": "glm_moe_dsa",
            "tp": 8,
            "ep": 1,
            "conc": 32,
            "isl": 1024,
            "osl": 1024,
            "image_digest": "rocm-ai/vllm:glm53-int4-gfx90a-0918",
            "rocm_version": "7.2.3",
            "kernel_optimizations": KERNEL_OPTIMIZATIONS,
            "status": "active",
            "status_note": (
                "在役基线（不是收敛值）：32k 档可用、事实召回 6/6；256k 待测、1M 未达成。"
                "对照 roofline T_mem(mi250x) 859 tok/s 目前只有 6.9% ⇒ 主要杠杆尚未收回，"
                "未验证项与已知边界见 remaining_gaps。QR 三条 env 一律留空（未启用），"
                "split-K 默认关（MI250_SPARSE_SPLITK=0）。"
            ),
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="", help="KB root（默认 $KNOWLEDGE_LOCAL_ROOT 或 <repo>/hyperloom/kb）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.root:
        root = Path(args.root)
    else:
        import os

        env_root = os.environ.get("KNOWLEDGE_LOCAL_ROOT", "").strip()
        root = Path(env_root) if env_root else REPO / "hyperloom" / "kb"

    r = row()
    if args.dry_run:
        print(json.dumps(r, ensure_ascii=False, indent=2, sort_keys=True))
        print("DRY: would put_recipe ->", root / "…" / CID.split(":")[1])
        return 0

    store = LocalRecipeStore(root=root)
    res = store.put_recipe(**r)
    c = res["counts"]
    print("put v%s %s %s" % (res["version"], "NEW" if res["created"] else "UPD", res["canonical_id"]))
    print("  counts: " + " ".join("%s=%s" % (k, v) for k, v in sorted(c.items())))
    print("  KB root:", root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
