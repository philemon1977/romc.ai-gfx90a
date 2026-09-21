#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把本机 gfx90a(MI250) 的实测硬约束写进 AMD 官方技能 serving-llms-on-instinct 的数据表（幂等）。

为什么需要：该技能直接读 `data/gpu_overrides.json > gpu_configs`（按 gfx 架构分档），
而它只覆盖 gfx942/gfx950（MI300/325/350/355）—— 全技能内 MI250/gfx90a 命中 0 次。
于是任何落在本机的任务，它会按通用/MI300 口径建议，而我们知道至少 5 条在这台机器上是反的。
第三方文件不入库（bootstrap 会还原），所以本脚本入库、**改完要重跑**（幂等，可反复跑）。

用法：python3 hyperloom/patches-local/apply_serving_skill_mi250x.py [--revert] [--check]
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path

SKILL = Path("/home/qiba/ROCm.AI/amd-skills/skills/serving-llms-on-instinct")
DATA = SKILL / "data"
OVR = DATA / "gpu_overrides.json"
BL = DATA / "blacklist.json"

GFX90A = {
    "env": {
        "VLLM_ROCM_USE_AITER_MOE": "0",
        "DSV41_IDX_AITER_KERNEL": "1",
        "MI250_SPARSE_SPLITK": "0",
    },
    "notes": [
        "MI250/gfx90a 实测（2026-09-21，GLM-5.3-CT-Int4-W4A16 TP8）：AITER 的 MoE 路径不可用；",
        "QuickReduce 禁用 —— init_custom_qr 固定吃 ~9 GiB/卡，而本模型 KV 总预算只有 8.17 GiB",
        "（VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB 压不动它）⇒ 启动显存门必死；",
        "sparse split-K 默认必须 0（conc 1/8/32 三档全负：-18%/-14%/-13%）；",
        "ENFORCE_EAGER=0 需同时给 MAX_CUDAGRAPH_CAPTURE_SIZE>=1，否则 vLLM 断言拒绝；",
        "DSV41_IDX_AITER_KERNEL=1 是实测 +69.3%（conc32）且默认值是本机不可信的 torch 回退；",
        "SKU 级口径：64 GiB/GCD、104 CU/GCD、TP8 时权重 52.9 GiB/rank ⇒ 32k 档 KV 仅 8.17 GiB/94k tokens；",
        "DCP=8 不省显存（权重 +2.11 GiB/rank）但把每 token KV 单价降到 1/8（93.4→11.5 KiB）。",
    ],
    "env_defaults": {
        "VLLM_ROCM_USE_AITER_MOE": "0",
        "DSV41_IDX_AITER_KERNEL": "1",
        "MI250_SPARSE_SPLITK": "0",
        "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "(不设=禁用；本机禁用 QR)",
    },
    "gpu_family": "CDNA2 (gfx90a)",
    "precision": "bf16 计算；权重 int4(W4A16)；KV 可 fp8_e4m3 存储（无需 FP8 矩阵核）。无原生 FP8/FP4 计算。",
    "vram_gb_note": "64 GiB/GCD x 8 GCD（104 CU/GCD）。勿按 MI250X=128 GiB 算 KV —— 那是两个 GCD 之和。TP8 时权重 52.9 GiB/rank，32k 档 KV 仅 8.17 GiB / 94,016 tokens。",
    "workarounds": [
        "AITER MoE 路径不可用（gfx90a 无该路径）",
        "QuickReduce 禁用：init_custom_qr 固定吃 ~9 GiB/卡 ≈ KV 全部预算，MAX_SIZE_BYTES_MB 压不动",
        "sparse split-K 默认 0：conc 1/8/32 三档全负（-18%/-14%/-13%）",
        "ENFORCE_EAGER=0 需同时给 MAX_CUDAGRAPH_CAPTURE_SIZE>=1，否则 vLLM 断言拒绝",
        "indexer decode 必须 DSV41_IDX_AITER_KERNEL=1（默认回退本机不可信，开启 +69.3% @conc32）",
        "DCP=8 不省显存（权重 +2.11 GiB/rank）但 KV 单价降到 1/8（93.4→11.5 KiB/token）",
    ],
}
BL_ENTRY = {
    "model_pattern": "GLM-5.3-CT-Int4-W4A16",
    "gpu": "gfx90a",
    "reason": "QuickReduce 与本次 KV 预算互斥（固定 ~9 GiB/卡 vs 8.17 GiB），启用即起不来；见 quark-int8/launcher 与 reports/models/glm53-int4/decode-config-ablation.md",
}


def backup(p: Path) -> None:
    b = p.with_suffix(p.suffix + ".pre-mi250x")
    if not b.exists():
        shutil.copyfile(p, b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not OVR.is_file():
        print("找不到 %s —— 技能未 bootstrap？" % OVR, file=sys.stderr)
        return 2
    if a.check:
        d = json.loads(OVR.read_text(encoding="utf-8"))
        ok = "gfx90a" in (d.get("gpu_configs") or {})
        print("gfx90a 档位存在:", ok)
        return 0 if ok else 1
    if a.revert:
        for p in (OVR, BL):
            b = p.with_suffix(p.suffix + ".pre-mi250x")
            if b.exists():
                shutil.copyfile(b, p)
                print("reverted", p.name)
        return 0
    changed = []
    backup(OVR)
    d = json.loads(OVR.read_text(encoding="utf-8"))
    cfgs = d.setdefault("gpu_configs", {})
    ref = cfgs.get("gfx942") or {}
    entry = dict(GFX90A)
    for k, v in ref.items():
        if isinstance(v, dict) and k not in entry:
            entry[k] = {}
        elif isinstance(v, list) and k not in entry:
            entry[k] = []
        elif isinstance(v, str) and k not in entry:
            entry[k] = ""
    if cfgs.get("gfx90a") != entry:
        cfgs["gfx90a"] = entry
        OVR.write_text(json.dumps(d, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")
        changed.append("gpu_configs.gfx90a")
    if BL.is_file():
        backup(BL)
        bl = json.loads(BL.read_text(encoding="utf-8"))
        if isinstance(bl, list):
            if not any(isinstance(x, dict) and x.get("model_pattern") == BL_ENTRY["model_pattern"] for x in bl):
                bl.append(BL_ENTRY)
                BL.write_text(json.dumps(bl, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")
                changed.append("blacklist += 1")
        elif isinstance(bl, dict):
            key = BL_ENTRY["model_pattern"]
            if bl.get(key) != BL_ENTRY:
                bl[key] = BL_ENTRY
                BL.write_text(json.dumps(bl, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")
                changed.append("blacklist[%s]" % key)
        else:
            print("blacklist 结构未知，跳过（type=%s）" % type(bl).__name__)
    # 功能断言：重新加载并核对 env（不是"文件写了就算"）
    d2 = json.loads(OVR.read_text(encoding="utf-8"))
    env2 = (d2.get("gpu_configs") or {}).get("gfx90a", {}).get("env", {})
    for k, v in GFX90A["env"].items():
        assert env2.get(k) == v, "断言失败：%s=%r 期望 %r" % (k, env2.get(k), v)
    print("applied:", ", ".join(changed) if changed else "无变化（幂等）")
    print("断言通过：gfx90a.env =", env2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())