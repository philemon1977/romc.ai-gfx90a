#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 MI250X 支持落到 Hyperloom（第三方代码，故以幂等 applier 形式固化）。

为什么是 applier 而不是 .patch：本机没有 pristine 的 Hyperloom 源树可 diff
（wheel 以 --target 装在工作区，且 bootstrap/install.sh 可能覆盖），所以用
"缺什么补什么、可重复执行"的方式表达，语义与 patch 等价且可自愈。

四处改动（每处都带来源注释，见 2026-09-21 的 hyperloom-mi250x-support-plan.md）：
  1) common/gpu_identity.py        加 ("mi250x", ("gfx90a", 104)) 身份行
  2) inference_optimizer/gpu_types.py   确保 _gpu_runner_type **不**把 mi250x 折叠到 mi300x
     （Magpie 侧的 vllm_mi250x.sh 由 apply_mi250x_runner.py 打进容器）
  3) inference_optimizer/gpu_types.py   _GFX_TO_RUNNER["gfx90a"] = "mi250x"（torch 探测兜底）
  4) orchestrator/kernel/roofline_ceiling.py   _MI250X_PEAK_TFLOPS + HW_SPECS["mi250x"]

用法：python3 apply_mi250x_identity.py [--install-dir /home/qiba/ROCm.AI/hyperloom]
"""
import argparse
import pathlib
import re
import sys

IDENT_ROW = '    "mi250x": ("gfx90a", 104),'
# 2026-09-21 晚：runner 不再折叠——Magpie 侧已补 vllm_mi250x.sh（见 apply_mi250x_runner.py），
# 所以 mi250x 是一个**真实的** runner 标签。这一步把早先的折叠改回去（幂等：已是目标态则跳过）。
FOLD_OLD = '    if normalized in ("mi325x", "mi308x", "mi250x"):'
FOLD_NEW = '    if normalized in ("mi325x", "mi308x"):'
ARCH_ROW = '    "gfx90a": "mi250x",   # MI250X：每 GCD 一个 device（本机 8 GCD = 8 device）'
PEAK_BLOCK = [
    '_MI250X_PEAK_TFLOPS: dict[str, float] = {',
    '    # 每 GCD（= 每 torch device）口径，与 num_gpus 语义一致（tensor-parallel degree）。',
    '    # 推导（2026-09-21 本机实测）：104 CU x 64 lane x 2 x 1.7 GHz = 22.63 TFLOP/s (fp32)；',
    '    # fp16/bf16 = 8 x fp32（CDNA2 MFMA 倍率，厂商 OAM 标称 383/47.9 = 8）。',
    '    # gfx90a 无原生 fp8/fp4 ⇒ 不列键，缺失时回退 T_mem。',
    '    "fp32": 22.6,',
    '    "float32": 22.6,',
    '    "bf16": 181.0,',
    '    "bfloat16": 181.0,',
    '    "fp16": 181.0,',
    '    "float16": 181.0,',
    '}',
    '',
]
SPEC_BLOCK = [
    '    "mi250x": {',
    '        # 每 GCD：64 GiB HBM2e、约 1.638 TB/s（OAM 3.28 TB/s 的一半）。',
    '        # 必须与 num_gpus 同口径，否则 T_mem 差 2 倍。',
    '        "hbm_gb": 64.0,',
    '        "hbm_bw_gbps": 1638.0,',
    '        "peak_tflops": _MI250X_PEAK_TFLOPS,',
    '    },',
]


def patch(path, changes):
    # changes: (label, needle, replacement, marker) — marker 用于"是否已打过"判定
    text = path.read_text()
    out = text
    done = []
    for label, needle, replacement, marker in changes:
        if marker and marker in out:
            done.append((label, "already"))
            continue
        if needle not in out:
            done.append((label, "MISSING ANCHOR"))
            continue
        out = out.replace(needle, replacement, 1)
        done.append((label, "applied"))
    if out != text:
        path.write_text(out)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install-dir", default="/home/qiba/ROCm.AI/hyperloom")
    a = ap.parse_args()
    root = pathlib.Path(a.install_dir)
    pkg = root / "hyperloom"
    if not pkg.is_dir():
        sys.exit("找不到 " + str(pkg))

    results = []
    results += patch(pkg / "common" / "gpu_identity.py", [
        ("身份行", "    \"mi300x\": (\"gfx942\", 304),", IDENT_ROW + '\n    "mi300x": ("gfx942", 304),', '"mi250x": ("gfx90a", 104)'),
    ])
    results += patch(pkg / "inference_optimizer" / "gpu_types.py", [
        ("runner 不折叠", FOLD_OLD, FOLD_NEW, '("mi325x", "mi308x")'),
        ("arch 兜底", '    "gfx942": "mi300x",', ARCH_ROW + '\n    "gfx942": "mi300x",', '"gfx90a": "mi250x"'),
    ])
    results += patch(pkg / "orchestrator" / "kernel" / "roofline_ceiling.py", [
        ("峰值表", "HW_SPECS: dict[str, dict[str, Any]] = {", "\n".join(PEAK_BLOCK) + "HW_SPECS: dict[str, dict[str, Any]] = {", '_MI250X_PEAK_TFLOPS: dict[str, float] = {'),
        ("HW_SPECS 行", 'HW_SPECS: dict[str, dict[str, Any]] = {\n    "mi300x": {', "HW_SPECS: dict[str, dict[str, Any]] = {\n" + "\n".join(SPEC_BLOCK) + '\n    "mi300x": {', '        "peak_tflops": _MI250X_PEAK_TFLOPS,'),
    ])
    bad = 0
    for label, status in results:
        print("  %-12s %s" % (label, status))
        if status == "MISSING ANCHOR":
            bad += 1
    print("OK" if bad == 0 else "有 %d 处锚点未命中，请人工检查" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())