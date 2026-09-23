#!/bin/bash
# 离线单步 profile（2026-09-20）：本版本 vLLM 不认 VLLM_TORCH_PROFILER_DIR（没有 /start_profile 路由），
# 所以按 launcher 的挂载/env 起一次性容器，在进程内用 torch.profiler 包住一次 generate。
# 用法：CTX=8192 GEN=32 bash quark-int8/scripts_local/glm_prof.sh       # 图模式（与在线一致）
#       EAGER=1 CTX=8192 bash quark-int8/scripts_local/glm_prof.sh      # eager（kernel 明细更干净）
set -o pipefail
[ -n "$AI_HOME" ] || AI_HOME=/home/qiba/ai
[ -n "$MODEL_PATH" ] || MODEL_PATH=/mnt/kioxia-cm6-3t8/ai/models/GLM/GLM-5.3-753B/CT-Int4-W4A16
[ -n "$IMG" ] || IMG=rocm-ai/vllm:glm53-int4-gfx90a-0918
[ -n "$CTX" ] || CTX=8192
[ -n "$GEN" ] || GEN=32
[ -n "$EAGER" ] || EAGER=0
[ -n "$ROWS" ] || ROWS=28
[ -n "$MOE_TUNED_DIR" ] || MOE_TUNED_DIR=/home/qiba/ai/config/moe-tuned
PATCH_ROOT="${AI_HOME}/recipes/patches/vllm/vllm-openai-rocm-nightly-0918/core/tree"
GEMV_PATCH="${AI_HOME}/recipes/patches/vllm/vllm-openai-rocm-nightly-0918/core/moe_gemv"
DP=/usr/local/lib/python3.12/dist-packages/vllm

# 显存硬门：等释放（最多 5 分钟）。注意 stop 容器后显存释放有滞后，实测会滞后数秒到数十秒。
for i in $(seq 1 60); do
  BUSY=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{if ($NF/1073741824 > 5) n++} END{print n+0}')
  [ "$BUSY" = "0" ] && break
  echo "   ⏳ 还有 $BUSY 个 GCD 占用 > 5 GiB，等 5s（第 $i 次）"
  sleep 5
done
if [ "$BUSY" != "0" ]; then
  echo "❌ 有 $BUSY 个 GCD 显存占用 > 5 GiB ⇒ 等释放，不抢卡"; exit 1
fi
[ -n "$KTRACE" ] || KTRACE=0
[ -n "$MI250_DCP" ] || MI250_DCP=8
[ -n "$MI250_PROF_WORKER" ] || MI250_PROF_WORKER=0
[ -n "$MI250_PROF_SKIP" ] || MI250_PROF_SKIP=10
[ -n "$MI250_PROF_STEPS" ] || MI250_PROF_STEPS=8
[ -n "$MI250_PROF_OUT" ] || MI250_PROF_OUT=/work/prof
mkdir -p /home/qiba/ROCm.AI/quark-int8/ktrace
if [ "$KTRACE" = "1" ]; then
  EP=(--entrypoint /opt/rocm/bin/rocprofv3)
  CMD=(--kernel-trace -f csv -o /work/ktrace/out -- python3 /work/prof_step.py --ctx "$CTX" --gen "$GEN" --eager "$EAGER")
  echo "   [rocprofv3] kernel-trace → quark-int8/ktrace/"
else
  EP=(--entrypoint python3)
  CMD=(/work/prof_step.py --ctx "$CTX" --gen "$GEN" --eager "$EAGER" --rows "$ROWS")
fi
echo "== prof: ctx=$CTX gen=$GEN eager=$EAGER dcp=$MI250_DCP ktrace=$KTRACE $(date +%T) =="
exec docker run --rm --name glm-prof "${EP[@]}" \
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  --group-add video --ipc=host --shm-size=16g \
  -v "$MODEL_PATH:/models:ro" \
  -v "$PATCH_ROOT/v1/attention/backends/mla/rocm_aiter_mla_sparse.py:$DP/v1/attention/backends/mla/rocm_aiter_mla_sparse.py:ro" \
  -v "$PATCH_ROOT/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:$DP/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:ro" \
  -v "$PATCH_ROOT/_aiter_ops.py:$DP/_aiter_ops.py:ro" \
  -v "$PATCH_ROOT/v1/attention/ops/rocm_aiter_mla_sparse.py:$DP/v1/attention/ops/rocm_aiter_mla_sparse.py:ro" \
  -v "$PATCH_ROOT/model_executor/layers/sparse_attn_indexer.py:$DP/model_executor/layers/sparse_attn_indexer.py:ro" \
  -v "$PATCH_ROOT/model_executor/model_loader/weight_utils.py:$DP/model_executor/model_loader/weight_utils.py:ro" \
  -v "$GEMV_PATCH:/patches/moe_gemv:ro" \
  -v "$MOE_TUNED_DIR:/moe-tuned:ro" \
  -v /home/qiba/ROCm.AI/quark-int8:/work \
  -e MI250_DCP="$MI250_DCP" \
  -e MI250_PROF_WORKER="$MI250_PROF_WORKER" \
  -e MI250_PROF_SKIP="$MI250_PROF_SKIP" \
  -e MI250_PROF_STEPS="$MI250_PROF_STEPS" \
  -e MI250_PROF_OUT="$MI250_PROF_OUT" \
  -e MI250_SPARSE_SPLITK="${MI250_SPARSE_SPLITK:-}" \
  -e MI250_SPARSE_DBG="${MI250_SPARSE_DBG:-}" \
  -e MI250_SPARSE_DBG_N="${MI250_SPARSE_DBG_N:-60}" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e DSV41_IDX_AITER_KERNEL=1 \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  -e AITER_TRITON_LOG_LEVEL=ERROR \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e HF_HUB_OFFLINE=1 -e VLLM_DISABLE_COMPILE_CACHE=1 \
  -e VLLM_ROCM_USE_AITER=0 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e VLLM_TUNED_CONFIG_FOLDER=/moe-tuned \
  -e FASTSAFETENSORS_ODIRECT=1 -e MI250_FST_MAX_BATCH_MB=2560 \
  -e PYTHONPATH=/patches/moe_gemv -e MI250_MOE_GEMV=1 \
  -e MI250_MOE_GEMV_MODULE=mi250_moe_gemv_gs \
  "$IMG" "${CMD[@]}"
