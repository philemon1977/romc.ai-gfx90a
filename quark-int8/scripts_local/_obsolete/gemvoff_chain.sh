#!/bin/bash
cd /home/qiba/ROCm.AI/quark-int8
# 先停掉注定 OOM 的 UVA 臂
PIDF=/home/qiba/ai/logs/dsv41ctint4-8119.pid
[ -f "$PIDF" ] && kill -TERM -"$(cat $PIDF)" 2>/dev/null
sleep 10; docker rm -f dsv41-ct-int4 >/dev/null 2>&1
sleep 50
echo "== 起：allbf16 仓（注意力 bf16 + 专家 int4 g32）+ MI250_MOE_GEMV=0（不挂我们的 GEMV）+ ENG_SKIP =="
export MODEL_PATH=/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4-opt-allbf16
export DSV41_ENG_SKIP=1 MI250_MOE_GEMV=0
export GPU_MEM_UTIL=0.97 MAX_MODEL_LEN=4096 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=1024 MAX_CUDAGRAPH_CAPTURE_SIZE=0
export ENFORCE_EAGER=1
D=/home/qiba/ai/models/deepseek-ai/launcher/deepseek_v4.1_flash_ctint4w4a16_vllm_rocmnightly0918_64k_8119_dsv41_mi250dx8.sh
bash "$D" > /tmp/gemvoff_launch.log 2>&1
grep -E "已后台启动|❌" /tmp/gemvoff_launch.log | head -3
sleep 20
L=$(readlink -f /home/qiba/ai/logs/dsv41ctint4/server-8119.current)
echo "LOG=$L"
for i in $(seq 1 70); do
  grep -qa "Application startup complete" "$L" && { echo "READY $(date +%T)"; break; }
  if grep -qaE "No available memory|OutOfMemory|EngineCore failed|Traceback .most recent" "$L"; then
    echo "FAILED $(date +%T)"; grep -aE "Available KV|No available memory|OutOfMemory|Error" "$L" | grep -av site-packages | tail -5 | cut -c1-230; exit 1
  fi
  docker ps -q -f name=dsv41-ct-int4 | grep -q . || { echo "CONTAINER GONE"; tail -25 "$L"; exit 1; }
  sleep 25
done
echo "=== 装载证据（关键：MoE 后端应显示 TRITON WNA16、且无 GEMV 补丁行）==="
grep -aE "Model loading took|GPU KV cache size|WNA16 MoE backend|kernel module = mi250" "$L" | tail -5 | cut -c1-185
echo "=== 事实召回（注意力 bf16 + 专家 int4 + 无 GEMV）==="
python3 fact_recall_probe.py 8119 /models
echo "=== 冒烟 ==="
curl -s -m 300 http://127.0.0.1:8119/v1/completions -H 'Content-Type: application/json' \
 -d '{"model":"/models","prompt":"What is 17*19? Answer: 17*19 = 323. So 15*15 =","max_tokens":32,"temperature":0}' | python3 -c "import json,sys;print(repr(json.load(sys.stdin)['choices'][0]['text']))"
echo "=== 停服还卡 ==="
[ -f "$PIDF" ] && kill -TERM -"$(cat $PIDF)" 2>/dev/null
sleep 15; docker rm -f dsv41-ct-int4 >/dev/null 2>&1
rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{printf "%.2f ", $NF/1073741824} END{print "GiB"}'
echo "GEMVOFF_ARM_DONE $(date +%T)"
