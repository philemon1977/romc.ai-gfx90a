#!/bin/bash
docker run --rm --entrypoint bash --cpus=6 -v /home/qiba/ROCm.AI:/w -v /tmp/dsv41_dump:/dump \
  -v /mnt/stripe-3mix-3t2/models/deepseek-ai/DeepSeek-V4.1-Flash:/src:ro \
  -v /mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4:/models:ro \
  -v /mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4-opt-allbf16:/new:ro \
  vllm/vllm-openai-rocm:nightly -c "
echo '===== layer2 MoE 损伤（旧 vs 新） ====='
nice -n 15 timeout 900 python3 /w/quark-int8/ref_moe_end2end.py --dump /dump/dsv41_L2_moe.pt --model /new --layer 2 2>&1 | tail -7
echo '===== layer2 注意力损伤 ====='
nice -n 15 timeout 900 python3 /w/quark-int8/quant_damage_attn.py 2 2>&1 | tail -10"
