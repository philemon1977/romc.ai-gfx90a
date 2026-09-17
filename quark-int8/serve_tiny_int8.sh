#!/usr/bin/env bash
# Serve the tiny Qwen3_5Moe INT8 (quark) checkpoint on one GCD with a very small
# memory footprint -- validates the whole INT8 integration path in ~1 minute.
set -euo pipefail
NAME=${NAME:-ornith-tiny-int8}
MODEL=${MODEL:-/work/tiny_int8}
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network host \
  --device /dev/kfd --device /dev/dri --group-add video \
  --shm-size 8G --ulimit memlock=-1:-1 \
  -e HIP_VISIBLE_DEVICES=${GPU:-0} \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=900 \
  -v /home/qiba/ROCm.AI/quark-int8:/work \
  -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 \
  vllm/vllm-openai-rocm:nightly \
    "$MODEL" --served-model-name tiny-int8 --port 8123 \
    --tensor-parallel-size 1 --gpu-memory-utilization ${MEM_UTIL:-0.05} \
    --max-model-len 1024 --max-num-seqs 4 --enforce-eager \
    --trust-remote-code
sleep 5; docker logs --tail 5 "$NAME"
