#!/bin/bash
# 量「新策略」对 layer0 的功能损伤（新目录分片3 = 试点产物，已含新策略）
docker run --rm --entrypoint bash \
  -v /home/qiba/ROCm.AI:/w -v /tmp/dsv41_dump:/dump \
  -v /mnt/stripe-3mix-3t2/models/deepseek-ai/DeepSeek-V4.1-Flash:/src:ro \
  -v /mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4:/models:ro \
  -v /mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4-opt-shbf16:/new:ro \
  vllm/vllm-openai-rocm:nightly -c "python3 /w/quark-int8/quant_damage.py --layer 0"
