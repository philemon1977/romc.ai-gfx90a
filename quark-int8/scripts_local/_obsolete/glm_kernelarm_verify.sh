#!/bin/bash
# 第二臂：DSV41_IDX_AITER_KERNEL=1（自研 gfx90a decode indexer 内核）
# 目的：与第一臂（默认=上游 torch 回退，不可信）对比 16K 针尖
set -u
R=/home/qiba/ROCm.AI/quark-int8
LOG=$R/logs/glm53_kernelarm_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== 第二臂（自研 decode 内核）$(date +%T) ==="
PORT=8121 MAX_MODEL_LEN=32768 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=4096 GPU_MEM_UTIL=0.97 \
ENFORCE_EAGER=1 DSV41_IDX_AITER_KERNEL=1 MI250_INDEXER_LOGITS_DEBUG=1 \
  bash /home/qiba/ai/models/ZhipuAI/launcher/glm53_int4w4a16_vllm_rocmnightly0918_32k_8121_mi250dx8.sh || exit 1
L=$(ls -t /home/qiba/ai/logs/glm53/server-8121-*.log | head -1); echo "SERVER_LOG=$L"
for i in $(seq 1 150); do
  grep -qa "Application startup complete" "$L" 2>/dev/null && { echo "READY $(date +%T)"; break; }
  grep -qaE "ValueError|AttributeError|RuntimeError|OutOfMemoryError" "$L" 2>/dev/null && {
    echo "FAILED $(date +%T)"; grep -aE "ValueError|AttributeError|RuntimeError|OutOfMemoryError" "$L" | tail -3 | cut -c1-200; break; }
  docker ps -q -f name=glm53-int4 | grep -q . || { echo "CONTAINER GONE"; tail -12 "$L" | cut -c1-200; break; }
  sleep 20
done
if grep -qa "Application startup complete" "$L" 2>/dev/null; then
  sleep 15
  echo "== indexer 路径取证 =="
  grep -a "DSV41\] indexer logits 路径" "$L" | tr "\r" "\n" | sort -u | head -2 | cut -c1-190
  echo "== 16K 针尖（关键判据）=="
  python3 $R/agent_bench.py --port 8121 --model glm-5.3 --doc-tokens 8 --turns 1 --answer-tokens 8 --needle-tokens 16384
  echo "== 8K 针尖（小一号对照）=="
  python3 $R/agent_bench.py --port 8121 --model glm-5.3 --doc-tokens 8 --turns 1 --answer-tokens 8 --needle-tokens 8192
  echo "== 事实召回 =="
  python3 $R/fact_recall_probe.py 8121 glm-5.3
  echo "== 崩溃扫描（应为 0）=="
  grep -ac "OutOfMemoryError" "$L"
fi
echo "KERNELARM_DONE $(date +%T)"
