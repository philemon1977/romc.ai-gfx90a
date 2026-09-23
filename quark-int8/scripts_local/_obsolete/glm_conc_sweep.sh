#!/bin/bash
set -u
cd /home/qiba/ROCm.AI
EAGER="${EAGER:-0}"; CG="${CG:-8}"; MBT="${MBT:-2048}"; MNS="${MNS:-32}"; CONC="${CONC:-1,4,8,16,32}"
echo "=== 并发扫描 eager=$EAGER cg=$CG mbt=$MBT max_num_seqs=$MNS conc=$CONC $(date +%T) ==="
docker rm -f glm53-int4 >/dev/null 2>&1 || true
for i in $(seq 1 40); do B=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{if ($NF/1073741824 > 5) n++} END{print n+0}'); [ "$B" = "0" ] && break; sleep 5; done
# DCP_SIZE 可外部指定（默认 8）；DCP=0 表示完全不开 DCP（单流对比用）
ENFORCE_EAGER="$EAGER" MAX_CUDAGRAPH_CAPTURE_SIZE="$CG" MAX_NUM_BATCHED_TOKENS="$MBT" MAX_NUM_SEQS="$MNS" DCP_SIZE="${DCP_SIZE:-8}" MAX_MODEL_LEN=32768 FORCE=1 PORT=8122 \
  bash quark-int8/scripts_local/glm_dcp_boot.sh 2>&1 | tail -3
for i in $(seq 1 80); do
  if docker logs glm53-int4 2>&1 | grep -qa "Application startup complete"; then echo "READY $(date +%T)"; break; fi
  docker ps -q -f name=glm53-int4 | grep -q . || { echo "CONTAINER_GONE $(date +%T)"; docker logs glm53-int4 2>&1 | tail -10 | cut -c1-170; exit 1; }
  sleep 10
done
sleep 8
docker logs glm53-int4 2>&1 | grep -aE "GPU KV cache size|Available KV" | tr "\r" "\n" | tail -2 | cut -c1-160
python3 -u quark-int8/speed_probe.py --port 8122 --lens 512,2048 --gen 64 --conc "$CONC"
echo "=== 并发扫描 DONE $(date +%T) ==="
