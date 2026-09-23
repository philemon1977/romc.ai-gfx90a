#!/bin/bash
# 就地修复 -opt-allbf16 仓的 FP4 专家 nibble 序，然后独立复验 + 清理临时件
set -u
R=/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4-opt-allbf16
echo "== 修复开始 $(date +%T) =="
cp -f $R/model.safetensors.index.json /tmp/index_allbf16_before.json
python3 /home/qiba/ROCm.AI/quark-int8/fix_fp4_nibble_order.py $R --apply 2>&1 | tail -6
echo "== 修复结束 $(date +%T) =="
echo "== 独立复验（源 vs 修复后） =="
docker run --rm --entrypoint bash \
 -v /home/qiba/ROCm.AI/quark-int8:/work \
 -v /mnt/stripe-3mix-3t2/models/deepseek-ai/DeepSeek-V4.1-Flash:/src:ro \
 -v /mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4-opt-allbf16:/ours:ro \
 vllm/vllm-openai-rocm:nightly-0918 -c 'cd /work && python3 -u audit_fp4_nibble_order.py /src /ours' 2>&1 | tail -14
echo "== 清理临时拷贝 =="
rm -rf /mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/.fixtest
df -h /mnt/kioxia-cm6-3t8 | tail -1
echo "FIX_JOB_DONE $(date +%T)"
