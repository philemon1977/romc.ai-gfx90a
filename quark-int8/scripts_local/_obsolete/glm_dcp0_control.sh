#!/bin/bash
# 对照实验：DCP=0（不开 DCP）@32K，跑事实召回 —— 判定乱码是 DCP 引起还是补丁本身引起
set -u
cd /home/qiba/ROCm.AI
echo "=== 对照 DCP=0 @32K 起服 $(date +%T) ==="
docker rm -f glm53-int4 >/dev/null 2>&1 || true
for i in $(seq 1 40); do B=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{if ($NF/1073741824 > 5) n++} END{print n+0}'); [ "$B" = "0" ] && break; sleep 5; done
DCP_SIZE=0 MAX_MODEL_LEN=32768 FORCE=1 PORT=8122 ENFORCE_EAGER=1 bash quark-int8/scripts_local/glm_dcp_boot.sh 2>&1 | tail -3
for i in $(seq 1 60); do
  if docker logs glm53-int4 2>&1 | grep -qa "Application startup complete"; then echo "READY $(date +%T)"; break; fi
  docker ps -q -f name=glm53-int4 | grep -q . || { echo "CONTAINER_GONE $(date +%T)"; exit 1; }
  sleep 10
done
sleep 8
docker logs glm53-int4 2>&1 | grep -aE "Model loading took|GPU KV cache size" | tr "\r" "\n" | tail -2 | cut -c1-160
echo "=== 事实召回（DCP=0 对照）==="
python3 -u quark-int8/fact_recall_probe.py 8122 glm-5.3
echo "=== 对照结束 $(date +%T) ==="
