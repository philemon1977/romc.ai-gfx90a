#!/bin/bash
cd /home/qiba/ROCm.AI/quark-int8
export MODEL_PATH=/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-MXFP4-ENGRAM4
export EXTRA_VOL=/mnt/stripe-3mix-3t2
export MAX_NUM_BATCHED_TOKENS=1024 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128
export MAX_MODEL_LEN=4096 MAX_NUM_SEQS=4 GPU_MEM_UTIL=0.95 MAX_CUDAGRAPH_CAPTURE_SIZE=0
export MI250_MOE_GEMV=0 ENFORCE_EAGER=1
export AITER_TRITON_LOG_LEVEL=ERROR VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_AITER_MOE=0
# 关键：不走 ENG_SKIP，改用 UVA offload 把 engram 表放宿主 RAM（按参数名筛选）
export VLLM_EXTRA_ARGS="--cpu-offload-gb 12 --cpu-offload-params engram.embed"
D=/home/qiba/ai/models/deepseek-ai/launcher/deepseek_v4.1_flash_OFFICIALENC_mxfp4_8119_mi250dx8.sh
bash "$D" > /tmp/ram_launch.log 2>&1
grep -E "已后台启动|❌" /tmp/ram_launch.log | head -3
sleep 20
L=$(readlink -f /home/qiba/ai/logs/dsv41ctint4/server-8119.current)
echo "LOG=$L"
for i in $(seq 1 70); do
  grep -qa "Application startup complete" "$L" && { echo "READY $(date +%T)"; break; }
  if grep -qaE "No available memory|OutOfMemory|EngineCore failed|Traceback .most recent" "$L"; then
    echo "FAILED $(date +%T)"; grep -aE "Available KV|No available memory|OutOfMemory|Error|offload" "$L" | grep -av site-packages | tail -6 | cut -c1-230; exit 1
  fi
  docker ps -q -f name=dsv41-ct-int4 | grep -q . || { echo "CONTAINER GONE"; tail -25 "$L"; exit 1; }
  sleep 25
done
echo "=== 装载与 offload 证据 ==="
grep -aE "Model loading took|GPU KV cache size|offload|Offload" "$L" | tail -6 | cut -c1-190
echo "=== 事实召回（官方编码 + engram 放 RAM）==="
python3 fact_recall_probe.py 8119 /models
echo "=== 冒烟 17*19 ==="
curl -s -m 300 http://127.0.0.1:8119/v1/completions -H 'Content-Type: application/json' \
 -d '{"model":"/models","prompt":"What is 17*19? Answer: 17*19 = 323. So 15*15 =","max_tokens":32,"temperature":0}' | python3 -c "import json,sys;print(repr(json.load(sys.stdin)['choices'][0]['text']))"
echo "=== 停服还卡 ==="
PIDF=/home/qiba/ai/logs/dsv41ctint4-8119.pid
[ -f "$PIDF" ] && kill -TERM -"$(cat $PIDF)" 2>/dev/null
sleep 15; docker rm -f dsv41-ct-int4 >/dev/null 2>&1
rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{printf "%.2f ", $NF/1073741824} END{print "GiB"}'
free -g | awk 'NR==2{print "host used=" $3 " avail=" $7 " GiB"}'
echo "RAM_ENGRAM_DONE $(date +%T)"
