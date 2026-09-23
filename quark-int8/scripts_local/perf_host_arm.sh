#!/bin/bash
# engram int4 表 -> 主机 pinned 内存（DSV41_ENG_HOST=1，装载直写宿主缓冲）+ cudagraph + TTFT/TPS
set -u
BASH_SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$BASH_SOURCE_DIR/arm_teardown_guard.sh"
bash "$BASH_SOURCE_DIR/arm_preflight.sh" "$"
PIDF=/home/qiba/ai/logs/dsv41ctint4-8119.pid
LOG=/home/qiba/ROCm.AI/quark-int8/logs/perf_host_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== 表在主机RAM 臂 $(date +%T) EAGER=${ENFORCE_EAGER:-0} ==="
export MODEL_PATH=/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4-opt-allbf16
unset DSV41_ENG_SKIP
unset DSV41_ENG_HOST 2>/dev/null || true
export DSV41_ENG_HOST_PREALLOC=1   # 表从一开始就建在主机 pinned 内存：显存付 0、无需释放（避免 ROCm 释放/GC 引发 GPU Hang）
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.95} MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192} MAX_NUM_SEQS=8 MAX_NUM_BATCHED_TOKENS=2048
export ENFORCE_EAGER=${ENFORCE_EAGER:-0} MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-256}
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}
# 关键：关掉 expandable_segments，否则 empty_cache 不把释放块还给驱动 ⇒ KV 算成负数
# 分配器保持启动器默认（expandable_segments）；本方案不做释放，无需改动
ISL_LIST=${ISL_LIST:-128,1024,4096}
unset VLLM_EXTRA_ARGS 2>/dev/null || true
D=/home/qiba/ai/models/deepseek-ai/launcher/deepseek_v4.1_flash_ctint4w4a16_vllm_rocmnightly0918_64k_8119_dsv41_mi250dx8.sh
T0=$(date +%s)
bash "$D" > /tmp/perf_host_launch.log 2>&1
grep -E "已后台启动|❌" /tmp/perf_host_launch.log | head -3
sleep 20
L=$(readlink -f /home/qiba/ai/logs/dsv41ctint4/server-8119.current)
echo "SERVER_LOG=$L"
for i in $(seq 1 90); do
  grep -qa "Application startup complete" "$L" && { echo "READY $(date +%T) 耗时 $(( $(date +%s) - T0 ))s"; break; }
  grep -qaE "hipErrorStreamCaptureUnsupported|No available memory|EngineCore failed" "$L" && { echo "FAILED $(date +%T)"; grep -aE "hipErrorStream|Available KV|No available memory" "$L" | tail -3 | cut -c1-190; break; }
  docker ps -q -f name=dsv41-ct-int4 | grep -q . || { echo "CONTAINER GONE $(date +%T)"; tail -8 "$L" | cut -c1-190; break; }
  sleep 20
done
if grep -qa "Application startup complete" "$L"; then
  echo "=== 关键证据 ==="
  grep -aE "DSV41_ENG_HOST=1|Model loading took|Available KV|GPU KV cache size|Capturing CUDA graphs \(FULL\)" "$L" | tr "\r" "\n" | tail -8 | cut -c1-180
  echo "=== 事实召回 ==="; python3 /home/qiba/ROCm.AI/quark-int8/fact_recall_probe.py 8119 /models
  echo "=== TTFT / TPS ==="
  python3 /home/qiba/ROCm.AI/quark-int8/perf_bench.py --port 8119 --isl "$ISL_LIST" \
     --max-tokens 16 --conc 1,4,8 --conc-isl 512 --conc-tokens 64 --label host \
     --out /home/qiba/ROCm.AI/quark-int8/logs/perf_host_$(date +%m%d_%H%M).json
fi
echo "=== 停服还卡（只在容器确属本轮时才动） ==="
if owns_container; then
  [ -f "$PIDF" ] && kill -TERM -"$(cat $PIDF)" 2>/dev/null
  sleep 15; docker rm -f dsv41-ct-int4 >/dev/null 2>&1
else
  echo "  容器不属于本轮（或无容器），跳过收尾，避免误杀"
fi
rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{printf "%.2f ", $NF/1073741824} END{print "GiB"}'
free -g | head -2
echo "HOST_ARM_DONE $(date +%T)"