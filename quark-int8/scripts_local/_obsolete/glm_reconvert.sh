#!/bin/bash
# 全量重转：indexer wk 改未量化(bf16) + 融合名 ignore
set -u
R=/home/qiba/ROCm.AI/quark-int8
O=/mnt/kioxia-cm6-3t8/ai/models/GLM/GLM-5.3-753B/CT-Int4-W4A16
LOG=$R/logs/glm53_reconvert_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== 全量重转 $(date +%T) ==="
rm -f $O/model-*.safetensors && echo "旧分片已清（$(ls $O/*.safetensors 2>/dev/null | wc -l) 个剩余）"
free -g | head -2; df -h /mnt/kioxia-cm6-3t8 | tail -1
docker run --rm --entrypoint bash --device /dev/kfd --device /dev/dri --group-add video \
 -v $R:/work -v /mnt/kioxia-cm6-3t8/ai/models/ZhipuAI:/mdl \
 vllm/vllm-openai-rocm:nightly-0918 -c "cd /work && python3 -u convert_glm53_ct_int4.py \
   --model /mdl/GLM-5.3 --out /mdl/GLM-5.3-CT-Int4-W4A16 --device cuda --device-budget-gib 12 --min-free-gib 4" 2>&1 | tail -12
sudo chown -R qiba:qiba $O 2>/dev/null; sudo chmod -R u+rwX,go+rX $O 2>/dev/null
echo "== 复审 $(date +%T) =="
docker run --rm --entrypoint bash -v $R:/work -v /mnt/kioxia-cm6-3t8/ai/models/ZhipuAI:/mdl \
 vllm/vllm-openai-rocm:nightly-0918 -c "cd /work && python3 -u audit_glm53_ct.py /mdl/GLM-5.3 /mdl/GLM-5.3-CT-Int4-W4A16" 2>&1 | tail -16
echo "RECONVERT_DONE $(date +%T)"
