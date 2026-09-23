#!/bin/bash
# 干跑：确认启动器的 AITER 开关能按环境变量展开（不真起服务）
set -u
L=/home/qiba/ai/models/deepseek-ai/launcher/deepseek_v4.1_flash_ctint4w4a16_vllm_rocmnightly0918_64k_8119_dsv41_mi250dx8.sh
bash -n "$L" && echo "启动器语法 OK"
sed -n "306,310p" "$L"
echo "--- 用 DRY_RUN 看最终 docker 参数里的 AITER 项 ---"
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=0 DRY_RUN=1 bash "$L" 2>&1 | grep -oE "VLLM_ROCM_USE_AITER[A-Z_]*=[01]" | sort -u
