#!/bin/bash
# 一次性把 GLM-5.3 int4 权重转成**按 rank 分片**的副本（每 rank 只读自己那 1/8）
# 目的：装载从"每个 rank 读 402 GB 的 mmap 乱序流（0.52 GB/s）"变成"每个 rank 读自己 50 GB
#       的顺序文件"，实测 8 路顺序流可到 4.67 GB/s ⇒ 装载预期 13 min → ~2 min。
# 依据：disk_diag 实测（单流 O_DIRECT 3.5 GB/s / 8 路 4.67 GB/s / 装载仅 0.52 GB/s）。
set -uo pipefail
AI_HOME=/home/qiba/ai
TREE=${AI_HOME}/recipes/patches/vllm/vllm-openai-rocm-nightly-0918/core/tree
GEMV=$AI_HOME/recipes/patches/gfx90a/ct_w4a16_dsv41_n0918/moe_gemv
MODEL="${MODEL:-/mnt/kioxia-cm6-3t8/ai/models/ZhipuAI/GLM-5.3-CT-Int4-W4A16}"
OUT_ROOT=/mnt/stripe-3mix-3t2/ai/sharded
OUT="$OUT_ROOT/GLM-5.3-CT-Int4-W4A16-TP8"
IMAGE=vllm/vllm-openai-rocm:nightly-0918
LOG_DIR=$AI_HOME/logs/glm53; mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/shardconv-$(date +%Y%m%d-%H%M%S).log

fail() { echo "❌ $*" >&2; exit 1; }
[ -f "$MODEL/config.json" ] || fail "模型不在：$MODEL"
mkdir -p "$OUT_ROOT" || fail "无法创建输出目录 $OUT_ROOT"
AVAIL_G=$(df -BG --output=avail "$OUT_ROOT" | tail -1 | tr -dc "0-9")
echo "输出卷可用 ${AVAIL_G}G（需要 ~420G）"
[ "${AVAIL_G:-0}" -ge 430 ] || fail "空间不足：$OUT_ROOT"
BUSY=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{if ($NF/1073741824 > 5) n++} END{print n+0}')
[ "${BUSY:-0}" -gt 0 ] && fail "有 $BUSY 个 GCD 显存占用 >5 GiB ⇒ 转换需要独占 8 卡，拒绝开跑"
docker rm -f glm53-int4 glm53-shardconv >/dev/null 2>&1 || true

echo "== 开始转换 $(date +%T) =="
echo "   源 $MODEL"
echo "   目标 $OUT （ZFS 条带）"
docker run -d --name glm53-shardconv --network host --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc host --shm-size 16g \
  -v "$MODEL":/models:ro -v "$OUT":/out \
  -v "$TREE/v1/attention/backends/mla/rocm_aiter_mla_sparse.py":/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py:ro \
  -v "$TREE/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py":/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:ro \
  -v "$TREE/_aiter_ops.py":/usr/local/lib/python3.12/dist-packages/vllm/_aiter_ops.py:ro \
  -v "$TREE/v1/attention/ops/rocm_aiter_mla_sparse.py":/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:ro \
  -v "$TREE/model_executor/layers/sparse_attn_indexer.py":/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py:ro \
  -v "$GEMV":/patches/moe_gemv:ro -v "$AI_HOME/config/moe-tuned":/moe-tuned:ro \
  -e PYTHONPATH=/patches/moe_gemv -e MI250_MOE_GEMV=1 -e MI250_MOE_GEMV_MODULE=mi250_moe_gemv_gs \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True -e AITER_TRITON_LOG_LEVEL=ERROR \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e HF_HUB_OFFLINE=1 -e VLLM_DISABLE_COMPILE_CACHE=1 \
  -e VLLM_ROCM_USE_AITER=0 -e VLLM_ROCM_USE_AITER_MOE=0 -e VLLM_TUNED_CONFIG_FOLDER=/moe-tuned \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  --entrypoint python3 "$IMAGE" \
  /app/vllm/examples/features/sharded_state/save_sharded_state_offline.py \
  --model /models --tensor-parallel-size 8 --output /out --load-format safetensors \
  --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.90 --enforce-eager --dtype bfloat16 \
  --file-pattern "model-rank-{rank}-part-{part}.safetensors" --max-file-size 5368709120 > /dev/null
sleep 8
CID=$(docker ps -q -f name=glm53-shardconv)
[ -n "$CID" ] || { echo "❌ 容器没起来"; docker logs glm53-shardconv 2>&1 | tail -20; exit 1; }
echo "   container=$CID"
echo "   日志：docker logs -f glm53-shardconv   （外部 tee: $LOG）"
docker logs -f glm53-shardconv > "$LOG" 2>&1 &
echo "   已开始跟踪日志到 $LOG"
