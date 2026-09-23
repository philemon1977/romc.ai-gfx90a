#!/bin/bash
# 最终交付配置 + TTFT/TPS 压测：修复后仓、engram 开、全补丁、GEMV=1
set -u
PIDF=/home/qiba/ai/logs/dsv41ctint4-8119.pid
LBL=${1:-eager}
LOG=/home/qiba/ROCm.AI/quark-int8/logs/perf_${LBL}_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== 性能臂 [$LBL] $(date +%T) ==="
export MODEL_PATH=/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4-opt-allbf16
unset DSV41_ENG_SKIP
export GPU_MEM_UTIL=0.95 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=8 MAX_NUM_BATCHED_TOKENS=2048
export ENFORCE_EAGER=${ENFORCE_EAGER:-1} MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-0}
D=/home/qiba/ai/models/deepseek-ai/launcher/deepseek_v4.1_flash_ctint4w4a16_vllm_rocmnightly0918_64k_8119_dsv41_mi250dx8.sh
bash "$D" > /tmp/perf_launch_${LBL}.log 2>&1
grep -E "已后台启动|❌" /tmp/perf_launch_${LBL}.log | head -3
sleep 20
L=$(readlink -f /home/qiba/ai/logs/dsv41ctint4/server-8119.current)
echo "SERVER_LOG=$L"
for i in $(seq 1 90); do
  grep -qa "Application startup complete" "$L" && { echo "READY $(date +%T)"; break; }
  grep -qaE "No available memory|OutOfMemory|EngineCore failed" "$L" && { echo "FAILED"; grep -aE "Available KV|No available memory|Error" "$L" | tail -4 | cut -c1-190; break; }
  docker ps -q -f name=dsv41-ct-int4 | grep -q . || { echo "CONTAINER GONE"; tail -15 "$L" | cut -c1-190; break; }
  sleep 20
done
if grep -qa "Application startup complete" "$L"; then
  echo "=== 硬件加速取证：内核/后端选择 ==="
  grep -aE "TritonW4A16LinearKernel|WNA16 MoE backend|kernel module = mi250|aiter|AITER|MLA" "$L" | head -8 | cut -c1-190
  grep -aE "Model loading took|GPU KV cache size" "$L" | tail -2 | cut -c1-175
  echo "=== TTFT / TPS ==="
  python3 /home/qiba/ROCm.AI/quark-int8/perf_bench.py --port 8119 --isl 128,1024,4096 \
     --max-tokens 16 --conc 1,4,8 --conc-isl 512 --conc-tokens 64 --label "$LBL" \
     --out /home/qiba/ROCm.AI/quark-int8/logs/perf_${LBL}_$(date +%m%d_%H%M).json
fi
echo "=== 停服还卡 ==="
[ -f "$PIDF" ] && kill -TERM -"$(cat $PIDF)" 2>/dev/null
sleep 15; docker rm -f dsv41-ct-int4 >/dev/null 2>&1
rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{printf "%.2f ", $NF/1073741824} END{print "GiB"}'
echo "PERF_ARM_${LBL}_DONE $(date +%T)"
