#!/bin/bash
# 等官方臂彻底结束（容器消失 + 显存释放）→ 跑 eager 性能臂 → 再跑 cudagraph 性能臂
set -u
echo "等待官方臂结束 $(date +%T)"
while docker ps -q -f name=dsv41-ct-int4 | grep -q .; do sleep 20; done
sleep 30
for i in $(seq 1 30); do
  U=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{s+=$NF} END{print s+0}')
  [ "$U" -lt 4294967296 ] && break
  sleep 20
done
echo "显存已释放，起 eager 臂 $(date +%T)"
bash /tmp/perf_arm.sh eager
sleep 40
echo "起 cudagraph 臂 $(date +%T)"
ENFORCE_EAGER=0 MAX_CUDAGRAPH_CAPTURE_SIZE=256 bash /tmp/perf_arm.sh cudagraph
echo "PERF_CHAIN_DONE $(date +%T)"
