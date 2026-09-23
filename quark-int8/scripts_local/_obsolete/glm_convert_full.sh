#!/bin/bash
# GLM-5.3 → compressed-tensors INT4 W4A16 全量转换（GPU，可续跑）
set -u
LOG=/home/qiba/ROCm.AI/quark-int8/logs/glm53_convert_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== GLM-5.3 → CT INT4 全量转换 $(date +%T) ==="
free -g | head -2; df -h /mnt/kioxia-cm6-3t8 | tail -1
docker run --rm --entrypoint bash --device /dev/kfd --device /dev/dri --group-add video \
 -v /home/qiba/ROCm.AI/quark-int8:/work \
 -v /mnt/kioxia-cm6-3t8/ai/models/ZhipuAI:/mdl \
 vllm/vllm-openai-rocm:nightly-0918 -c "cd /work && python3 -u convert_glm53_ct_int4.py \
   --model /mdl/GLM-5.3 --out /mdl/GLM-5.3-CT-Int4-W4A16 \
   --device cuda --skip-existing --device-budget-gib 12 --min-free-gib 4"
echo "CONVERT_EXIT=$?"
echo "=== 产物 ==="; du -sh /mnt/kioxia-cm6-3t8/ai/models/GLM/GLM-5.3-753B/CT-Int4-W4A16 2>/dev/null
ls /mnt/kioxia-cm6-3t8/ai/models/GLM/GLM-5.3-753B/CT-Int4-W4A16/*.safetensors 2>/dev/null | wc -l
echo "GLM53_CONVERT_DONE $(date +%T)"
