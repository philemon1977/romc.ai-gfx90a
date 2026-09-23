#!/bin/bash
# Desc: 验证烘进镜像的两处修复 + 官方解析器 flag（一次起服答两件事，跑完即停）
set -uo pipefail
L="${1:?}"
OUT=/home/qiba/ai/logs/glm53flash-0918/baked-$(date +%H%M%S).log
for i in $(seq 1 120); do
  grep -qa 'Application startup complete' "$L" && { echo "READY $((i*10))s"; break; }
  grep -qaE 'exited gracefully|EngineDead|unrecognized arguments' "$L" && { echo '启动失败：'; grep -aE 'unrecognized arguments|only supported on AITER|error' "$L" | tail -4 | cut -c1-160; exit 1; }
  sleep 10
done
{
echo '### A. 镜像自给自足的证明（零挂载）'
grep -acE 'only supported on AITER' $L | sed 's/^/  indexer 门报错次数(期望0)=/'
grep -ac 'TileLang' $L | sed 's/^/  TileLang 编译行数(期望0，镜像内 mhc 已排除 gfx90a)=/'
grep -aoE "'block_size': [0-9]+" $L | head -1 | sed 's/^/  /'
grep -aoE 'Using [A-Z_]+ backend|TRITON Int8 MoE' $L | sort -u | sed 's/^/  /'
grep -acE 'reasoning_parser|tool_call_parser' $L | sed 's/^/  配置里出现解析器字段次数=/'
echo '### B. 解析器行为的实测'
python3 /tmp/verify_parsers.py
echo '### C. NLL 量级（同一次 boot）'
python3 /home/qiba/ai/tools/probe_nll.py --url http://127.0.0.1:8128/v1 --model glm53flash-int8 2>&1 | tail -9
} 2>&1 | tee "$OUT"
echo "DONE $OUT"