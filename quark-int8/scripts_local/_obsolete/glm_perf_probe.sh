#!/bin/bash
# TPS 瓶颈定位：并发扩展性 + torch profiler（保留服务，不自动停）
set -u
R=/home/qiba/ROCm.AI/quark-int8
LOG=$R/logs/glm53_perf_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== TPS 瓶颈定位 $(date +%T) ==="
PORT=8121 MAX_MODEL_LEN=32768 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=4096 GPU_MEM_UTIL=0.97 \
ENFORCE_EAGER=1 PROF_DIR=/home/qiba/ai/logs/glm53/prof \
  bash /home/qiba/ai/models/ZhipuAI/launcher/glm53_int4w4a16_vllm_rocmnightly0918_32k_8121_mi250dx8.sh || exit 1
echo "SERVER_LOG=$(ls -t /home/qiba/ai/logs/glm53/server-8121-*.log | head -1)"
# ★ 等待一律以 docker logs 为准（上一轮用 ls -t 挑文件被 0 字节文件骗到，白等）
for i in $(seq 1 150); do
  docker logs glm53-int4 2>&1 | grep -qa "Application startup complete" && { echo "READY $(date +%T)"; break; }
  docker ps -q -f name=glm53-int4 | grep -q . || { echo "CONTAINER GONE $(date +%T)"; break; }
  sleep 20
done
if docker logs glm53-int4 2>&1 | grep -qa "Application startup complete"; then
  sleep 15
  echo "== 加速项取证 =="
  docker logs glm53-int4 2>&1 | tr '\r' '\n' | grep -aE "TritonW4A16LinearKernel|WNA16 MoE backend|TritonWNA16Experts|MI250_MOE_GEMV|Using default MoE config|all-reduce backends|LBNHC" | sort -u | tail -8 | cut -c1-180
  echo "== 并发扩展 + profiler =="
  python3 $R/bench_profile.py --port 8121 --model glm-5.3 --out $R/logs/perf_report_$(date +%m%d_%H%M).json
fi
echo "PERF_DONE $(date +%T)  （服务保留）"
