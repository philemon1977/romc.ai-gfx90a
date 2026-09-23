#!/bin/bash
# GLM-5.3 int4 @32K 针尖验证（item 0）：16K 针尖 + 小规模多轮前缀复用
set -u
R=/home/qiba/ROCm.AI/quark-int8
LOG=/home/qiba/ROCm.AI/quark-int8/logs/glm53_needle_0920_1734.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 针尖验证开始 $(date +%T) （服务已在 8121，不重启，省 12.5 min 装载） ==="
echo "== agent 多轮（8K 文档 x3 轮）=="
python3 $R/agent_bench.py --port 8121 --model glm-5.3 --doc-tokens 8192 --turns 3 --answer-tokens 48
echo "== 8K 针尖（对照组：短上下文应先命中）=="
python3 $R/agent_bench.py --port 8121 --model glm-5.3 --doc-tokens 8 --turns 1 --answer-tokens 8 --needle-tokens 8192
echo "== 16K 针尖（关键：稀疏路径 + KV 布局）=="
python3 $R/agent_bench.py --port 8121 --model glm-5.3 --doc-tokens 8 --turns 1 --answer-tokens 8 --needle-tokens 16384
echo "NEEDLE_DONE $(date +%T)"
