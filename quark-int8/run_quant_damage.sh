#!/bin/bash
cd /home/qiba/ROCm.AI/quark-int8
docker run --rm --entrypoint bash \
  -v /home/qiba/ROCm.AI:/w -v /tmp/dsv41_dump:/dump \
  -v /mnt/stripe-3mix-3t2/models/deepseek-ai/DeepSeek-V4.1-Flash:/src:ro \
  -v /mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4:/models:ro \
  -v /mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4-optscale:/new:ro \
  vllm/vllm-openai-rocm:nightly -c "python3 /w/quark-int8/quant_damage.py --layer 0; echo '--- layer 2 ---'; python3 /w/quark-int8/quant_damage.py --layer 2 --dump /dump/dsv41_L2_moe.pt"
