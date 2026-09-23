#!/bin/bash
# 启动 UVA 臂 -> 等"卸载完成"后再静默 5 分钟 -> SIGABRT 两个 worker 抓全部 Python 线程栈
set -u
echo "=== 等 GPU 释放 ==="
for i in $(seq 1 30); do
  U=$(rocm-smi --showmeminfo vram 2>/dev/null | awk "/Used/{s+=\$NF} END{print s+0}")
  [ "$U" -lt 5368709120 ] && break
  sleep 10
done
echo "=== 起臂 $(date +%T) ==="
VLLM_ROCM_USE_AITER=0 ENFORCE_EAGER=0 MAX_CUDAGRAPH_CAPTURE_SIZE=256 MAX_MODEL_LEN=8192 GPU_MEM_UTIL=0.93 OFFLOAD_GB=13 ISL_LIST=128,1024,4096 nohup bash /tmp/perf_uva_arm.sh > /dev/null 2>&1 &
sleep 60
L=""
for i in $(seq 1 40); do
  L=$(readlink -f /home/qiba/ai/logs/dsv41ctint4/server-8119.current 2>/dev/null)
  [ -n "$L" ] && grep -qa "Total CPU offloaded parameters" "$L" && break
  sleep 15
done
echo "=== 卸载已发生，日志=$L $(date +%T) ==="
grep -a "Total CPU offloaded" "$L" | tail -1 | cut -c1-120
echo "=== 静默观察 5 分钟 ==="
for i in $(seq 1 10); do
  SZ=$(stat -c %s "$L"); echo "  $(date +%T) logsize=$SZ"
  sleep 30
done
echo "=== SIGABRT 抓栈 $(date +%T) ==="
for w in Worker_TP0 Worker_TP3; do
  P=$(docker exec dsv41-ct-int4 bash -c "pgrep -f $w | head -1" 2>/dev/null)
  [ -n "$P" ] && docker exec dsv41-ct-int4 bash -c "kill -ABRT $P" 2>/dev/null && echo "  已 ABRT $w (pid=$P)"
done
sleep 20
echo "=== 栈（日志尾部） ==="
tail -120 "$L" | grep -aE "Current thread|File \"|Thread 0x|line [0-9]+, in|wait|lock|sync|collective|broadcast" | head -60 | cut -c1-190
echo "=== 原文尾部 60 行 ==="
tail -60 "$L" | cut -c1-190
echo "ABORT_DONE $(date +%T)"
