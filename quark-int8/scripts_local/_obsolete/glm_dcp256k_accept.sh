#!/bin/bash
# DCP=8 @256K 完整验收（FST 快装载）：事实召回 + GSM8K + 针尖曲线 + parity 基线
set -u
cd /home/qiba/ROCm.AI
echo "=== DCP=8 @262144 验收 $(date +%T) ==="
docker rm -f glm53-int4 >/dev/null 2>&1 || true
for i in $(seq 1 40); do B=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{if ($NF/1073741824 > 5) n++} END{print n+0}'); [ "$B" = "0" ] && break; sleep 5; done
DCP_SIZE=8 MAX_MODEL_LEN=262144 FORCE=1 PORT=8122 ENFORCE_EAGER=1 bash quark-int8/scripts_local/glm_dcp_boot.sh 2>&1 | tail -3
for i in $(seq 1 80); do
  if docker logs glm53-int4 2>&1 | grep -qa "Application startup complete"; then echo "READY $(date +%T)"; break; fi
  docker ps -q -f name=glm53-int4 | grep -q . || { echo "CONTAINER_GONE $(date +%T)"; exit 1; }
  sleep 10
done
sleep 8
docker logs glm53-int4 2>&1 | grep -aE "Model loading took|GPU KV cache size|Available KV" | tr "\r" "\n" | tail -3 | cut -c1-170
echo "== 事实召回 =="; python3 -u quark-int8/fact_recall_probe.py 8122 glm-5.3
echo "== GSM8K =="; python3 -u quark-int8/glm_gsm8k_probe.py 8122 glm-5.3
echo "== 针尖曲线 =="; python3 -u quark-int8/agent_bench2.py --port 8122 --model glm-5.3 --sweep 2048,4096,8192,16384 --save-prompt /tmp/dcp256k_prompt.txt
echo "== parity 基线 =="; python3 -u quark-int8/dcp_patches/dcp_parity_probe.py --port 8122 --tokens 4096 --out quark-int8/logs/parity_dcp8_256k_$(date +%m%d_%H%M).json
echo "=== 256K 验收 DONE $(date +%T) ==="
