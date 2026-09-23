#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 gfx90a(MI250X) 上打开 QuickReduce（移植前人 C2+C3，2026-09-21）。

出处：/home/qiba/ai/recipes/patches/vllm/vllm_0.28.0_rocm72/port_p1_c2_c3.py（该文件开头即写"给 gfx90a(MI250X) 打开 QuickReduce"），
以及 /home/qiba/ai/recipes/patches/native-env/vllm028-base-p1-qr.patch。前人目标版本 vLLM 0.28，本仓镜像为
0.3.1.dev85；两处锚点在本镜像里**逐字相同**，故可原样移植。

C2  distributed/device_communicators/quick_all_reduce.py
      supported_archs = ["gfx94", "gfx95"]  ->  ["gfx90", "gfx94", "gfx95"]
      依据：FP(lossless) 模式在 CDNA2 可用且与 NCCL 逐位一致（前人实测）。
C3  distributed/device_communicators/cuda_communicator.py
      qr_comm 初始化不再要求 use_custom_allreduce（arch 门由 C2 管）。
      只让 qr_comm 单独上场，**不碰 ca_comm** —— CUSTOM 在 gfx90a 上静默算错（输出全 !!!）。

三条 env（缺 C3 就白设，应写进 launcher）：
    VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB=0    # 最关键：decode AR 仅 4-10KB
    VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=FP        # C3 的触发条件（!= NONE）
    VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0    # 默认 True 会改 dtype、污染基线

用法（容器内，需 root 写 dist-packages）：
    docker exec hyperloom-local python3 /home/qiba/ROCm.AI/hyperloom/patches-local/apply_gfx90a_quickreduce.py [--revert]
"""
import argparse
import os
import pathlib
import shutil
import sys

VLLM = os.environ.get("VLLM_ROOT", "/usr/local/lib/python3.12/dist-packages/vllm")
QR = pathlib.Path(VLLM) / "distributed" / "device_communicators" / "quick_all_reduce.py"
CC = pathlib.Path(VLLM) / "distributed" / "device_communicators" / "cuda_communicator.py"

QR_OLD = '            supported_archs = ["gfx94", "gfx95"]'
# 注意：每个元素必须带逗号！曾因三个相邻字面量漏逗号被 Python 隐式拼接成一行，
# 导致 supported_archs 赋值整行变成注释、QR 静默不启用（2026-09-21 实测踩到）。
QR_NEW = [
    '            # MI250X(gfx90a) C2: FP(lossless) 模式在 CDNA2 可用且与 NCCL 逐位一致；',
    '            # 只加 gfx90，其余量化模式在 gfx90a 上静默算错（前人实测，勿开）。',
    '            supported_archs = ["gfx90", "gfx94", "gfx95"]',
]
CC_OLD = "        if use_custom_allreduce and self.world_size > 1 and current_platform.is_rocm():"
CC_NEW = [
    "        # MI250X(gfx90a) C3: qr_comm 不再要求 use_custom_allreduce（arch 门由 C2 管），",
    "        # 且只让 qr_comm 上场、不碰 ca_comm —— CUSTOM 在 gfx90a 上静默算错。",
    "        if self.world_size > 1 and current_platform.is_rocm():",
]


def edit(path, old, new_lines, revert, tag):
    if not path.is_file():
        print("  %-4s 文件不在: %s" % (tag, path)); return 1
    text = path.read_text()
    new = "\n".join(new_lines)
    if revert:
        if new in text:
            path.write_text(text.replace(new, old, 1))
            print("  %-4s reverted" % tag)
        else:
            print("  %-4s already-clean" % tag)
        return 0
    if new in text:
        print("  %-4s already" % tag); return 0
    if old not in text:
        print("  %-4s MISSING ANCHOR（版本不符，需人工核对）" % tag); return 1
    bak = str(path) + ".pre-qr"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    path.write_text(text.replace(old, new, 1))
    print("  %-4s applied（备份 %s）" % (tag, os.path.basename(bak)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    # ★ 2026-09-21 前置门：本机实测 QR 对本模型不可用（固定 ~9 GiB/卡 vs KV 总预算 8.17 GiB）。
    # 打补丁本身没问题（代码层与 NCCL 逐位一致），问题在"启用之后"的显存账 ⇒ 默认拦住；
    # 确要在别的模型 / 更宽松显存上启用时，显式 QR_MEM_ACK=1 放行。
    if not a.revert and not os.environ.get("QR_MEM_ACK"):
        print("⚠️  QR 是本机已判死的路线（2026-09-21，四条臂全部死在启动显存门）：")
        print("    init_custom_qr 固定占 ~9 GiB/卡 ≈ GLM-5.3-int4 的 KV 全部预算（8.17 GiB），")
        print("    VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB 压不动它；不开 QR 时同刻 free 63.0 GiB。")
        print("    依据：hyperloom/reports/models/glm53-int4/decode-config-ablation.md 第三节")
        print("    要在别的模型/更宽松显存上启用：QR_MEM_ACK=1 重跑本脚本。")
        return 3
    bad = edit(QR, QR_OLD, QR_NEW, a.revert, "C2")
    bad += edit(CC, CC_OLD, CC_NEW, a.revert, "C3")
    import ast
    for p in (QR, CC):
        ast.parse(p.read_text())
    print("  语法 OK；环境变量请设（launcher/config/vllm.conf）：")
    for k, v in (("VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB", "0"),
                 ("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", "FP"),
                 ("VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", "0")):
        print("    %s=%s" % (k, v))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())