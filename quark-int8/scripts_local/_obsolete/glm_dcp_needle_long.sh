#!/bin/bash
# 长上下文针尖（256K 档）：等 256K 验收电池跑完再跑，避免并发污染 TTFT
set -u
cd /home/qiba/ROCm.AI
echo "=== 等电池结束 $(date +%T) ==="
for i in $(seq 1 120); do
  if ls -t quark-int8/logs/glm_dcp256k_accept_*.log 2>/dev/null | head -1 | xargs grep -qa "256K 验收 DONE" 2>/dev/null; then echo "电池已完成，开始长针尖 $(date +%T)"; break; fi
  sleep 15
done
echo "=== 长上下文针尖：32K / 128K / 245K（唯一句填充 + 数字码）==="
python3 -u quark-int8/agent_bench2.py --port 8122 --model glm-5.3 \
  --sweep 32768,131072,245760 --answer-tokens 24 --save-prompt /tmp/dcp_long_prompt.txt
echo "=== 长针尖 DONE $(date +%T) ==="
