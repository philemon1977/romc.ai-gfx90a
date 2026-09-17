#!/usr/bin/env bash
# Pilot: convert a single DSV4.1-Flash shard to compressed-tensors INT4 W4A16.
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
SRC=/mnt/stripe-3mix-3t2/models/deepseek-ai/DeepSeek-V4.1-Flash
OUT=${OUT:-$REPO/pilot_ct_int4}
SHARDS=${SHARDS:-3}
GPU=${GPU:-1}
exec docker run --rm --entrypoint bash --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 993 --user 1000:1000 -e HOME=/tmp \
  -e HIP_VISIBLE_DEVICES=$GPU -v /home/qiba/ROCm.AI:/w \
  -v /mnt/stripe-3mix-3t2:/mnt/stripe-3mix-3t2:ro \
  --workdir /w/quark-int8 vllm/vllm-openai-rocm:nightly \
  -c "python3 /w/quark-int8/convert_dsv41_ct_int4.py --model $SRC --out $OUT \
      --shards $SHARDS --device cuda"
