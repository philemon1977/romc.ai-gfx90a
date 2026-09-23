#!/bin/bash
# DCP 验收电池（①/④）：等 READY -> 事实召回 -> GSM8K -> 针尖曲线 -> 存 parity 基线
# 用法: PORT=8122 bash scripts_local/glm_dcp_accept.sh
set -u
R=/home/qiba/ROCm.AI/quark-int8
PORT="${PORT:-8122}"
LOG=$R/logs/glm_dcp_accept_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== DCP 验收电池 $(date +%T) port=$PORT ==="

# 等 READY（★ 以 docker logs 为准：ls -t 挑文件曾被 0 字节文件骗到，白等一轮）
for i in $(seq 1 120); do
  if docker logs glm53-int4 2>&1 | grep -qa "Application startup complete"; then echo "READY $(date +%T)"; break; fi
  docker ps -q -f name=glm53-int4 | grep -q . || { echo "CONTAINER_GONE $(date +%T)"; exit 1; }
  sleep 15
done
docker logs glm53-int4 2>&1 | grep -qa "Application startup complete" || { echo "NOT_READY_TIMEOUT"; exit 1; }
sleep 10

echo "== DCP 生效取证 =="
docker logs glm53-int4 2>&1 | tr "\r" "\n" | grep -aE "GPU KV cache size|Available KV cache memory|decode_context_parallel_size|max_model_len" | tail -5 | cut -c1-190

echo "== 事实召回 =="
python3 -u $R/fact_recall_probe.py $PORT glm-5.3
echo "== GSM8K =="
python3 -u $R/glm_gsm8k_probe.py $PORT glm-5.3
echo "== 针尖命中率曲线（唯一句填充 + 数字码 + 自校准）=="
python3 -u $R/agent_bench2.py --port $PORT --model glm-5.3 --sweep 2048,4096,8192,16384 --save-prompt /tmp/dcp_accept_prompt.txt
echo "== 存 parity 基线（供 DCP=1 对照）=="
python3 -u $R/dcp_patches/dcp_parity_probe.py --port $PORT --tokens 4096 --out $R/logs/parity_dcp8_$(date +%m%d_%H%M).json
echo "ACCEPT_DONE $(date +%T)  （服务保留，供后续对照）"
