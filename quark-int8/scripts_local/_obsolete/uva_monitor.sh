#!/bin/bash
# 重跑 UVA 臂 + 每 30s 采样进展（RSS/缺页/日志 mtime/显存），最多 30 分钟
set -u
VLLM_ROCM_USE_AITER=0 ENFORCE_EAGER=0 MAX_CUDAGRAPH_CAPTURE_SIZE=256 MAX_MODEL_LEN=8192 GPU_MEM_UTIL=0.93 OFFLOAD_GB=13 ISL_LIST=128,1024,4096 nohup bash /tmp/perf_uva_arm.sh > /dev/null 2>&1 &
sleep 90
for i in $(seq 1 60); do
  L=$(readlink -f /home/qiba/ai/logs/dsv41ctint4/server-8119.current 2>/dev/null)
  P=$(docker exec dsv41-ct-int4 bash -c "pgrep -f Worker_TP0 | head -1" 2>/dev/null || echo "")
  if [ -n "$P" ]; then
    S=$(docker exec dsv41-ct-int4 bash -c "awk '{print \$10, \$12}' /proc/$P/stat; grep VmRSS /proc/$P/status" 2>/dev/null | tr "\n" " ")
  else S="(no worker)"; fi
  MT=$(stat -c %y "$L" 2>/dev/null | cut -d. -f1)
  VR=$(rocm-smi --showmeminfo vram 2>/dev/null | awk "/Used/{s+=\$NF} END{printf \"%.0f\", s/1073741824}")
  RDY=$(grep -ac "Application startup complete" "$L" 2>/dev/null || echo 0)
  echo "$(date +%T) i=$i rss/flt[$S] log_mtime=$MT vram_total=${VR}GiB ready=$RDY"
  [ "$RDY" != "0" ] && { echo "READY"; break; }
  grep -qa "hipErrorStream\|OutOfMemory\|EngineCore failed" "$L" 2>/dev/null && { echo "FAILED"; grep -aE "hipError|OutOfMemory|Error" "$L" | tail -3 | cut -c1-160; break; }
  sleep 30
done
echo "MONITOR_DONE $(date +%T)"
