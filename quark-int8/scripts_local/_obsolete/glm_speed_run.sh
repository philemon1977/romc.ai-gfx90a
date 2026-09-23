#!/bin/bash
# 速度专项（可选 graph / MTP）：**先停旧容器**再起服（否则守卫端口门会拒绝），然后跑速度探针
set -u
cd /home/qiba/ROCm.AI
EAGER="${EAGER:-1}"; CG="${CG:-0}"; MBT="${MBT:-2048}"; LENS="${LENS:-512,1024,2048,4096}"; GEN="${GEN:-64}"; CONC="${CONC:-4}"
SPEC="${SPEC:-}"
echo "=== 速度专项 eager=$EAGER cg=$CG mbt=$MBT spec=${SPEC:-none} $(date +%T) ==="
docker rm -f glm53-int4 >/dev/null 2>&1 || true
for i in $(seq 1 40); do B=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{if ($NF/1073741824 > 5) n++} END{print n+0}'); [ "$B" = "0" ] && break; sleep 5; done
if [ -n "$SPEC" ]; then export SPEC_CONFIG="$SPEC"; fi
ENFORCE_EAGER="$EAGER" MAX_CUDAGRAPH_CAPTURE_SIZE="$CG" MAX_NUM_BATCHED_TOKENS="$MBT" DCP_SIZE=8 MAX_MODEL_LEN=32768 FORCE=1 PORT=8122 \
  bash quark-int8/scripts_local/glm_dcp_boot.sh 2>&1 | tail -3
for i in $(seq 1 80); do
  if docker logs glm53-int4 2>&1 | grep -qa "Application startup complete"; then echo "READY $(date +%T)"; break; fi
  docker ps -q -f name=glm53-int4 | grep -q . || { echo "CONTAINER_GONE $(date +%T)"; docker logs glm53-int4 2>&1 | tail -12 | cut -c1-180; exit 1; }
  sleep 10
done
sleep 8
docker logs glm53-int4 2>&1 | grep -aE "Model loading took|GPU KV cache size|Available KV|speculative|MTP|mtp" | tr "\r" "\n" | tail -4 | cut -c1-170
python3 -u quark-int8/speed_probe.py --port 8122 --lens "$LENS" --gen "$GEN" --conc "$CONC"
echo "=== 速度专项 DONE $(date +%T) ==="
