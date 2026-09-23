#!/bin/bash
# 内核级定位：起服（graph 模式 + PROF_DIR）→ /start_profile → 跑解码 → /stop_profile → 汇总
set -u
cd /home/qiba/ROCm.AI
PROF=/home/qiba/ai/logs/glm53/prof_kernel; mkdir -p $PROF; rm -f $PROF/*.json 2>/dev/null
echo "=== 起服（带 profiler 目录）$(date +%T) ==="
docker rm -f glm53-int4 >/dev/null 2>&1 || true
for i in $(seq 1 40); do B=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{if ($NF/1073741824 > 5) n++} END{print n+0}'); [ "$B" = "0" ] && break; sleep 5; done
PROF_DIR=$PROF EAGER=0 CG=8 MBT=2048 DCP_SIZE=8 MAX_MODEL_LEN=32768 FORCE=1 PORT=8122 \
  bash quark-int8/scripts_local/glm_dcp_boot.sh >/dev/null 2>&1 &
for i in $(seq 1 80); do
  if docker logs glm53-int4 2>&1 | grep -qa "Application startup complete"; then echo "READY $(date +%T)"; break; fi
  docker ps -q -f name=glm53-int4 | grep -q . || { echo "CONTAINER_GONE"; exit 1; }
  sleep 10
done
sleep 5
python3 - <<'PYEOF'
import json, time, urllib.request
def post(ep, body=None):
    req = urllib.request.Request("http://127.0.0.1:8122" + ep, data=(json.dumps(body).encode() if body else b""),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=120).read()[:120]
def gen(n=96):
    body = json.dumps({"model":"glm-5.3","prompt":"The quick brown fox "*200,"max_tokens":n,"temperature":0.0}).encode()
    req = urllib.request.Request("http://127.0.0.1:8122/v1/completions", body, {"Content-Type":"application/json"})
    urllib.request.urlopen(req, timeout=900).read()
print("warmup:", len(gen(32)))
print("start_profile:", post("/start_profile"))
for i in range(3):
    gen(64)
print("stop_profile:", post("/stop_profile"))
PYEOF
sleep 20
echo "=== trace 文件 ==="; ls -la $PROF | head -6
echo "=== 内核时间表（top 20）==="
python3 /home/qiba/ROCm.AI/quark-int8/scripts_local/prof_kernel_top.py $PROF 2>/dev/null | head -28
echo "=== DONE $(date +%T) ==="
