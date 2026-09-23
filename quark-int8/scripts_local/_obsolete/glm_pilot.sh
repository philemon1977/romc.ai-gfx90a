#!/bin/bash
set -u
cd /home/qiba/ROCm.AI/quark-int8
echo "== 试点：分片 1,2（CPU，避免抢正在跑的臂的显存） $(date +%T) =="
docker run --rm --entrypoint bash \
 -v /home/qiba/ROCm.AI/quark-int8:/work \
 -v /mnt/kioxia-cm6-3t8/ai/models/ZhipuAI:/mdl \
 vllm/vllm-openai-rocm:nightly-0918 -c "cd /work && python3 -u convert_glm53_ct_int4.py --model /mdl/GLM-5.3 --out /mdl/GLM-5.3-CT-Int4-W4A16-pilot --device cpu --shards 1,2"
echo "== 复验（独立尺子） $(date +%T) =="
docker run --rm --entrypoint bash \
 -v /home/qiba/ROCm.AI/quark-int8:/work \
 -v /mnt/kioxia-cm6-3t8/ai/models/ZhipuAI:/mdl \
 vllm/vllm-openai-rocm:nightly-0918 -c "cd /work && python3 -u verify_glm53_int4.py /mdl/GLM-5.3 /mdl/GLM-5.3-CT-Int4-W4A16-pilot"
echo "PILOT_DONE $(date +%T)"
