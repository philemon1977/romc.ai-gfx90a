#!/bin/bash
# 修复后仓 + engram 打开（int4 表驻显存）：最终交付配置的验收臂
set -u
PIDF=/home/qiba/ai/logs/dsv41ctint4-8119.pid
LOG=/home/qiba/ROCm.AI/quark-int8/logs/fixed_engram_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== 修复后 + engram 开 臂 $(date +%T) ==="
export MODEL_PATH=/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4-opt-allbf16
unset DSV41_ENG_SKIP
export GPU_MEM_UTIL=0.95 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=1024
D=/home/qiba/ai/models/deepseek-ai/launcher/deepseek_v4.1_flash_ctint4w4a16_vllm_rocmnightly0918_64k_8119_dsv41_mi250dx8.sh
bash "$D" > /tmp/fixed_engram_launch.log 2>&1
grep -E "已后台启动|❌" /tmp/fixed_engram_launch.log | head -3
sleep 20
L=$(readlink -f /home/qiba/ai/logs/dsv41ctint4/server-8119.current)
echo "SERVER_LOG=$L"
for i in $(seq 1 90); do
  grep -qa "Application startup complete" "$L" && { echo "READY $(date +%T)"; break; }
  grep -qaE "No available memory|OutOfMemory|EngineCore failed" "$L" && { echo "FAILED"; grep -aE "Available KV|No available memory|Error" "$L" | tail -4 | cut -c1-190; STOP=1; break; }
  docker ps -q -f name=dsv41-ct-int4 | grep -q . || { echo "CONTAINER GONE"; tail -15 "$L" | cut -c1-190; STOP=1; break; }
  sleep 20
done
grep -aE "Model loading took|GPU KV cache size|Available KV" "$L" | tail -4 | cut -c1-180
if grep -qa "Application startup complete" "$L"; then bash /tmp/arm_battery.sh engram; fi
echo "=== 停服还卡 ==="
[ -f "$PIDF" ] && kill -TERM -"$(cat $PIDF)" 2>/dev/null
sleep 15; docker rm -f dsv41-ct-int4 >/dev/null 2>&1
rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{printf "%.2f ", $NF/1073741824} END{print "GiB"}'
echo "ENGRAM_ARM_DONE $(date +%T)"
