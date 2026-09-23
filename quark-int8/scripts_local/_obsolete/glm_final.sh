#!/bin/bash
# GLM-5.3 INT4 最终验收：审计门 -> 1M 上下文起服（cudagraph）-> agent 场景 + 质量 + 性能 -> 停服
set -u
R=/home/qiba/ROCm.AI/quark-int8
LOG=$R/logs/glm53_final_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== GLM-5.3 最终验收 $(date +%T) ==="

echo "== 审计门 =="
OUT=$(docker run --rm --entrypoint bash -v $R:/work -v /mnt/kioxia-cm6-3t8/ai/models/ZhipuAI:/mdl \
  vllm/vllm-openai-rocm:nightly-0918 -c "cd /work && python3 -u audit_glm53_ct.py /mdl/GLM-5.3 /mdl/GLM-5.3-CT-Int4-W4A16" 2>&1 | tail -16)
echo "$OUT"; echo "$OUT" | grep -q "AUDIT: PASS" || { echo "❌ 审计未过"; exit 1; }

echo "== 起服（1M 上下文 + cudagraph + agent 并发配置）$(date +%T) =="
PORT=8121 MAX_MODEL_LEN=1048576 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=8192 \
  ENFORCE_EAGER=0 MAX_CUDAGRAPH_CAPTURE_SIZE=256 GPU_MEM_UTIL=0.97 \
  bash /home/qiba/ai/models/ZhipuAI/launcher/glm53_int4w4a16_vllm_rocmnightly0918_32k_8121_mi250dx8.sh || exit 1
L=$(ls -t /home/qiba/ai/logs/glm53/server-8121-*.log | head -1); echo "SERVER_LOG=$L"
for i in $(seq 1 180); do
  grep -qa "Application startup complete" "$L" 2>/dev/null && { echo "READY $(date +%T)"; break; }
  docker ps -q -f name=glm53-int4 | grep -q . || { echo "CONTAINER GONE"; tail -30 "$L" | cut -c1-200; exit 1; }
  sleep 20
done
if grep -qa "Application startup complete" "$L" 2>/dev/null; then
  echo "== 量化/加速/容量取证 =="
  grep -aE "CompressedTensorsWNA16|WNA16 MoE backend|kernel module = mi250|TritonW4A16LinearKernel|Model loading took|GPU KV cache size|Available KV|maximum concurrency" "$L" | tail -8 | cut -c1-190
  grep -aE "Capturing CUDA graphs \((PIECEWISE|FULL)\)" "$L" | tr "\r" "\n" | tail -2 | cut -c1-120
  echo "== 事实召回 =="; python3 $R/fact_recall_probe.py 8121 glm-5.3
  echo "== agent 多轮 + 128K 针尖 =="
  python3 $R/agent_bench.py --port 8121 --model glm-5.3 --doc-tokens 32768 --turns 4 --needle-tokens 131072 \
     2>&1 | tee $R/logs/glm53_agent_$(date +%m%d_%H%M).log
  echo "== ★ 1M 上下文针尖（目标验收项；预填充可能数分钟）=="
  python3 $R/agent_bench.py --port 8121 --model glm-5.3 --needle-tokens 1048576 --turns 1 --doc-tokens 512 \
     2>&1 | tee $R/logs/glm53_needle1m_$(date +%m%d_%H%M).log
  echo "== TTFT/TPS =="; python3 $R/perf_bench.py --port 8121 --model glm-5.3 --isl 128,4096,32768 --max-tokens 16 \
     --conc 1,4 --conc-isl 2048 --conc-tokens 64 --label glm53_final --out $R/logs/perf_glm53_final.json
fi
echo "== 停服 $(date +%T) =="; docker rm -f glm53-int4 >/dev/null 2>&1; sleep 10
rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{printf "%.1f ", $NF/1073741824} END{print "GiB"}'
echo "GLM53_FINAL_DONE $(date +%T)"