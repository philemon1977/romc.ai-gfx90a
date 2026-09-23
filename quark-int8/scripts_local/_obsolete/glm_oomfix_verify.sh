#!/bin/bash
# GLM-5.3 int4 @32K：验证 fp8_mqa_logits_torch 分块修复（长 prefill OOM 回归）
# 判据：① 8K 文档多轮不再崩（修复前 8 个 rank 同时 OOM）
#       ② 16K 针尖能出答案且命中
set -u
R=/home/qiba/ROCm.AI/quark-int8
LOG=$R/logs/glm53_oomfix_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== OOM 修复验证 $(date +%T) ==="
PORT=8121 MAX_MODEL_LEN=32768 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=4096 GPU_MEM_UTIL=0.97 \
ENFORCE_EAGER=1 MI250_INDEXER_LOGITS_DEBUG=1 \
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
  echo "== 取证（KV 容量 / 内核选择 / 后端）=="
  grep -aE "Model loading took|GPU KV cache size|maximum concurrency|WNA16 MoE backend|Using .*backend out of potential" "$L" | tr "\r" "\n" | sort -u | tail -6 | cut -c1-175
  echo "== ① 8K 文档多轮（修复前即在此崩：8 rank 同刻 OOM）=="
  python3 $R/agent_bench.py --port 8121 --model glm-5.3 --doc-tokens 8192 --turns 3 --answer-tokens 48
  echo "== ② 16K 针尖（item 0 的关键判据）=="
  python3 $R/agent_bench.py --port 8121 --model glm-5.3 --doc-tokens 8 --turns 1 --answer-tokens 8 --needle-tokens 16384
  echo "== ③ 事实召回（确认修复未伤质量）=="
  python3 $R/fact_recall_probe.py 8121 glm-5.3
  echo "== 分块取证（M/H/N 与块数）=="
  grep -a "MI250_IDX_LOGITS" "$L" | tr "\r" "\n" | sort -u | tail -12
  echo "== 崩溃扫描（应为 0）=="
  grep -ac "OutOfMemoryError" "$L"
fi
echo "== 保留服务以便继续验证；如需停：docker rm -f glm53-int4 （$(date +%T)）=="
echo "OOMFIX_DONE $(date +%T)"
