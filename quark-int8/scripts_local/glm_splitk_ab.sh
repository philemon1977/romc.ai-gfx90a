#!/bin/bash
# split-K A/B 单臂（2026-09-21）：同一配置下 MI250_SPARSE_SPLITK=$S，量 TPS + 事实召回。
# 邻居保护：不杀任何非本容器进程；显存用 GPU_MEM_UTIL=0.92（GPU2 上邻居占 2.99 GiB，留 2.1 GiB 余量）；
#           容器 --cpu-shares 256 让出 CPU；起服前后都对邻居 PID/显存取证。
set -uo pipefail
cd /home/qiba/ROCm.AI
S="${MI250_SPARSE_SPLITK:-0}"
PORT="${PORT:-8122}"
NAME=glm53-int4

NEIGH_PID=$(rocm-smi --showpids 2>/dev/null | grep -E "^[0-9]+" | awk '{print $1}' | sort -u | head -1)
echo "== arm S=${S} $(date +%T) =="
echo "   邻居 pid=${NEIGH_PID:-无}  $(ps -p ${NEIGH_PID:-0} -o etime=,args= --no-headers 2>/dev/null | cut -c1-70)"
rocm-smi --showmeminfo vram 2>/dev/null | grep -a "VRAM Total Used" | awk '{printf "   起服前 GPU%d %.2f GiB\n", NR-1, $NF/1073741824}'

GPU_MEM_UTIL=0.92 CPU_SHARES=256 MI250_SPARSE_SPLITK="${S}" \
DCP_SIZE=8 MAX_MODEL_LEN=32768 MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=32 \
ENFORCE_EAGER=0 MAX_CUDAGRAPH_CAPTURE_SIZE=8 PORT=${PORT} FORCE=1 \
  bash quark-int8/scripts_local/glm_dcp_boot.sh 2>&1 | tail -3

for i in $(seq 1 90); do
  if docker logs ${NAME} 2>&1 | grep -qa "Application startup complete"; then echo "READY $(date +%T)"; break; fi
  if ! docker ps -q -f name=${NAME} | grep -q .; then echo "CONTAINER_GONE"; docker logs ${NAME} 2>&1 | tail -10 | cut -c1-160; exit 1; fi
  sleep 10
done
docker logs ${NAME} 2>&1 | tr '\r' '\n' | grep -aE "Available KV|KV cache size" | tail -1 | cut -c1-150
echo "--- 事实召回"
python3 -u quark-int8/fact_recall_probe.py ${PORT} glm-5.3 2>&1 | tail -3
echo "--- 速度（lens 2048, gen 64）"
python3 -u quark-int8/speed_probe.py --port ${PORT} --lens 2048 --gen 64 --conc "1,4,8,16,32" 2>&1 | tail -8

echo "--- 邻居状态（应与起服前一致）"
if [ -n "${NEIGH_PID:-}" ]; then ps -p ${NEIGH_PID} -o pid,etime,args --no-headers 2>/dev/null | cut -c1-80 || echo "   ！邻居进程不在了"; fi
rocm-smi --showmeminfo vram 2>/dev/null | grep -a "VRAM Total Used" | awk '{printf "   停服前 GPU%d %.2f GiB\n", NR-1, $NF/1073741824}'
docker rm -f ${NAME} >/dev/null 2>&1
sleep 8
rocm-smi --showmeminfo vram 2>/dev/null | grep -a "VRAM Total Used" | awk '{printf "   停服后 GPU%d %.2f GiB\n", NR-1, $NF/1073741824}'
echo "== arm S=${S} DONE（已停服）=="
