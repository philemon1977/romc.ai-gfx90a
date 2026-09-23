#!/bin/bash
# cudagraph(+AITER) 性能臂：就绪 -> 内核取证 -> 事实召回(防图路径静默出错) -> TTFT/TPS -> 停服
set -u
PIDF=/home/qiba/ai/logs/dsv41ctint4-8119.pid
LBL=${1:-cg}
LOG=/home/qiba/ROCm.AI/quark-int8/logs/perf2_${LBL}_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== 性能臂2 [$LBL] $(date +%T)  EAGER=${ENFORCE_EAGER:-1} AITER=${VLLM_ROCM_USE_AITER:-0} ==="
export MODEL_PATH=/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4-opt-allbf16
unset DSV41_ENG_SKIP
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.95} MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192} MAX_NUM_SEQS=8 MAX_NUM_BATCHED_TOKENS=2048
ISL_LIST=${ISL_LIST:-128,1024,4096}
export ENFORCE_EAGER=${ENFORCE_EAGER:-0} MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-256}
D=/home/qiba/ai/models/deepseek-ai/launcher/deepseek_v4.1_flash_ctint4w4a16_vllm_rocmnightly0918_64k_8119_dsv41_mi250dx8.sh
bash "$D" > /tmp/perf2_launch_${LBL}.log 2>&1
grep -E "已后台启动|❌" /tmp/perf2_launch_${LBL}.log | head -3
sleep 20
L=$(readlink -f /home/qiba/ai/logs/dsv41ctint4/server-8119.current)
echo "SERVER_LOG=$L"
for i in $(seq 1 120); do
  grep -qa "Application startup complete" "$L" && { echo "READY $(date +%T)"; break; }
  grep -qaE "hipErrorStreamCaptureUnsupported|No available memory|OutOfMemory|EngineCore failed" "$L" && { echo "FAILED $(date +%T)"; grep -aE "hipErrorStream|operation not permitted|Available KV|OutOfMemory|Error" "$L" | tail -4 | cut -c1-200; break; }
  docker ps -q -f name=dsv41-ct-int4 | grep -q . || { echo "CONTAINER GONE"; tail -12 "$L" | cut -c1-200; break; }
  sleep 20
done
if grep -qa "Application startup complete" "$L"; then
  echo "=== 内核/后端取证 ==="
  grep -aE "Breakable CUDA graph|WNA16 MoE backend|kernel module = mi250|Graph capturing finished|TritonW4A16" "$L" | tail -5 | cut -c1-180
  grep -aE "Model loading took|GPU KV cache size" "$L" | tail -2 | cut -c1-175
  echo "=== 事实召回（cudagraph 下质量是否仍正常）==="
  python3 /home/qiba/ROCm.AI/quark-int8/fact_recall_probe.py 8119 /models
  echo "=== TTFT / TPS ==="
  python3 /home/qiba/ROCm.AI/quark-int8/perf_bench.py --port 8119 --isl "$ISL_LIST" \
     --max-tokens 16 --conc 1,4,8 --conc-isl 512 --conc-tokens 64 --label "$LBL" \
     --out /home/qiba/ROCm.AI/quark-int8/logs/perf_${LBL}_$(date +%m%d_%H%M).json
fi
echo "=== 停服还卡 ==="
[ -f "$PIDF" ] && kill -TERM -"$(cat $PIDF)" 2>/dev/null
sleep 15; docker rm -f dsv41-ct-int4 >/dev/null 2>&1
rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{printf "%.2f ", $NF/1073741824} END{print "GiB"}'
echo "PERF2_${LBL}_DONE $(date +%T)"
