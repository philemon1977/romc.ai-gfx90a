#!/bin/bash
# 官方编码臂（源 MXFP4 专家 + 官方 fp8 注意力，B 仓）+ 同一 battery
set -u
PIDF=/home/qiba/ai/logs/dsv41ctint4-8119.pid
LOG=/home/qiba/ROCm.AI/quark-int8/logs/official_battery_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== 官方编码 + battery 臂 $(date +%T) ==="
export MODEL_PATH=/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-MXFP4-ENGRAM4
export DSV41_ENG_SKIP=1 MI250_MOE_GEMV=0
export EXTRA_VOL=/mnt/stripe-3mix-3t2   # B 仓分片是指向源仓的符号链接，必须同路径挂进容器
export GPU_MEM_UTIL=0.97 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=1024
D=/home/qiba/ai/models/deepseek-ai/launcher/deepseek_v4.1_flash_OFFICIALENC_mxfp4_8119_mi250dx8.sh
bash "$D" > /tmp/official_battery_launch.log 2>&1
grep -E "已后台启动|❌" /tmp/official_battery_launch.log | head -3
sleep 20
L=$(readlink -f /home/qiba/ai/logs/dsv41ctint4/server-8119.current)
echo "SERVER_LOG=$L"
for i in $(seq 1 80); do
  grep -qa "Application startup complete" "$L" && { echo "READY $(date +%T)"; break; }
  grep -qaE "No available memory|OutOfMemory|EngineCore failed" "$L" && { echo "FAILED"; grep -aE "Available KV|No available memory|Error" "$L" | tail -4 | cut -c1-190; exit 1; }
  docker ps -q -f name=dsv41-ct-int4 | grep -q . || { echo "CONTAINER GONE"; tail -15 "$L" | cut -c1-190; exit 1; }
  sleep 20
done
grep -aE "Model loading took|GPU KV cache size" "$L" | tail -3 | cut -c1-175
bash /tmp/arm_battery.sh official
echo "=== 停服还卡 ==="
[ -f "$PIDF" ] && kill -TERM -"$(cat $PIDF)" 2>/dev/null
sleep 15; docker rm -f dsv41-ct-int4 >/dev/null 2>&1
rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{printf "%.2f ", $NF/1073741824} END{print "GiB"}'
echo "OFFICIAL_ARM_DONE $(date +%T)"
