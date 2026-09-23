#!/bin/bash
# GLM-5.3 int4 @32K 完整验证：起服(eager) -> 取证 -> 事实召回 -> GSM8K -> agent 多轮+针尖 -> 停服
set -u
R=/home/qiba/ROCm.AI/quark-int8
LOG=$R/logs/glm53_verify32k_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== GLM-5.3 int4 @32K 验证 $(date +%T) ==="
PORT=8121 MAX_MODEL_LEN=32768 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=4096 GPU_MEM_UTIL=0.97 ENFORCE_EAGER=1 \
  bash /home/qiba/ai/models/ZhipuAI/launcher/glm53_int4w4a16_vllm_rocmnightly0918_32k_8121_mi250dx8.sh || exit 1
L=$(ls -t /home/qiba/ai/logs/glm53/server-8121-*.log | head -1); echo "SERVER_LOG=$L"
for i in $(seq 1 150); do
  grep -qa "Application startup complete" "$L" 2>/dev/null && { echo "READY $(date +%T)"; break; }
  grep -qaE "ValueError|AttributeError|RuntimeError" "$L" 2>/dev/null && { echo "FAILED $(date +%T)"; grep -aE "ValueError|AttributeError|RuntimeError" "$L" | tail -3 | cut -c1-200; break; }
  docker ps -q -f name=glm53-int4 | grep -q . || { echo "CONTAINER GONE"; tail -12 "$L" | cut -c1-200; break; }
  sleep 20
done
if grep -qa "Application startup complete" "$L" 2>/dev/null; then
  sleep 15
  echo "== 取证 =="
  grep -aE "Model loading took|GPU KV cache size|maximum concurrency|WNA16 MoE backend" "$L" | tr "\r" "\n" | tail -4 | cut -c1-175
  echo "== 事实召回 =="; python3 $R/fact_recall_probe.py 8121 glm-5.3
  echo "== GSM8K =="; python3 $R/glm_gsm8k_probe.py 8121 glm-5.3
  echo "== agent 多轮 + 16K 针尖 =="
  python3 $R/agent_bench.py --port 8121 --model glm-5.3 --doc-tokens 8192 --turns 3 --answer-tokens 48 --needle-tokens 16384
fi
echo "== 停服 $(date +%T) =="; docker rm -f glm53-int4 >/dev/null 2>&1; sleep 8
rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{printf "%.1f ", $NF/1073741824} END{print "GiB"}'
echo "VERIFY32K_DONE $(date +%T)"
