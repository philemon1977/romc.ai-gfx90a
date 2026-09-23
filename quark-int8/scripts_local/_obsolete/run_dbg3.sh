#!/bin/bash
# gfx90a indexer aiter 路径：数值对拍 + 图捕获测试（后台，落盘）
set -u
P=${AI_HOME}/recipes/patches/vllm/vllm-openai-rocm-nightly-0918/core/tree
docker run --rm --entrypoint bash --device /dev/kfd --device /dev/dri --group-add video \
 -v /home/qiba/ROCm.AI/quark-int8:/work \
 -v "$P/v1/attention/ops/rocm_aiter_mla_sparse.py":/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:ro \
 -v "$P/_aiter_ops.py":/usr/local/lib/python3.12/dist-packages/vllm/_aiter_ops.py:ro \
 vllm/vllm-openai-rocm:nightly-0918 -c "cd /work && python3 -u idx_dbg3.py"
echo "PARITY_EXIT=$?"