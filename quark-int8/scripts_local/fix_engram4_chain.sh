#!/bin/bash
# 修复 -engram4 仓的 FP4 专家 nibble 序（就地、纯 I/O），并独立复验
set -u
R=/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4
echo "== 修复开始 $(date +%T)  $R =="
python3 /home/qiba/ROCm.AI/quark-int8/fix_fp4_nibble_order.py "$R" --apply 2>&1 | tail -4
echo "== 修复结束 $(date +%T) =="
docker run --rm --entrypoint bash \
 -v /home/qiba/ROCm.AI/quark-int8:/work \
 -v /mnt/stripe-3mix-3t2/models/deepseek-ai/DeepSeek-V4.1-Flash:/src:ro \
 -v "$R":/ours:ro \
 vllm/vllm-openai-rocm:nightly-0918 -c "cd /work && python3 -u audit_fp4_nibble_order.py /src /ours" 2>&1 | tail -12
echo "FIX_ENGRAM4_DONE $(date +%T)"
