#!/bin/bash
# 停服 → 等容器退出且显存释放 → 用 DSV41_PROFILE 重启（40 层健康画像 + 修好门的旋转链 dump）
set -u
L=/home/qiba/ai/models/deepseek-ai/launcher/deepseek_v4.1_flash_ctint4w4a16_vllm_rocmnightly_256k_8119_dsv41_mi250dx8.sh
LOG=/home/qiba/ROCm.AI/quark-int8/logs/v64_profile_$(date +%m%d_%H%M).log

echo "=== $(date +%T) 停服 ==="
docker stop -t 30 dsv41-ct-int4 2>&1 | tail -1

echo "=== $(date +%T) 等容器退出 + 显存释放 ==="
for i in $(seq 1 60); do
  sleep 10
  if docker ps --format '{{.Names}}' | grep -qx dsv41-ct-int4; then
    echo "  t=$((i*10))s 容器仍在"; continue
  fi
  MAX=$(for d in 0 1 2 3 4 5 6 7; do
    rocm-smi --showmeminfo vram -d "$d" 2>/dev/null | awk '/Used Memory/{print $NF}'
  done | sort -n | tail -1)
  MIB=$(( ${MAX:-0}/1048576 ))
  echo "  t=$((i*10))s 容器已退出 最大占用=${MIB} MiB"
  if [ "$MIB" -lt 2000 ]; then echo "✅ $(date +%T) 显存已释放"; break; fi
done

echo "=== $(date +%T) 重启（DSV41_PROFILE=1 DSV41_DUMP_DEBUG=1）→ $LOG ==="
DSV41_PROFILE=1 DSV41_DUMP_DEBUG=1 DSV41_FFN_DEBUG=1 setsid nohup bash "$L" > "$LOG" 2>&1 &
echo "launcher pid=$! log=$LOG"
sleep 30
grep -a "✅\|❌" "$LOG" | head -8
