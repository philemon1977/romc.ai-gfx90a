#!/bin/bash
OUT=$1
LIST=$(cd "$OUT" && ls model-000*-of-00048.safetensors 2>/dev/null | tr '\n' ' ')
docker run --rm --entrypoint bash --cpus=4 -v /home/qiba/ROCm.AI:/w \
  -v /mnt/stripe-3mix-3t2/models/deepseek-ai/DeepSeek-V4.1-Flash:/src:ro \
  -v "$OUT":/new:ro \
  vllm/vllm-openai-rocm:nightly -c "nice -n 19 timeout 1500 python3 /w/quark-int8/verify_shard_types.py $LIST"
