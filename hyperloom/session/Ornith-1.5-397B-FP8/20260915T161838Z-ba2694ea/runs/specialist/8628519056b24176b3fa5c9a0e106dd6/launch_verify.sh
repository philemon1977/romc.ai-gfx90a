#!/bin/bash
# Verification boot of Ornith-1.5-397B-FP8 on 8xMI250X with the STACKED
# enablement base (patch 001 gfx90a FP8->BF16 emulation + patch 002 AOT-cache
# invalidation) applied in an isolated worktree, on cards that are now FREE.
#
# Deliberately uses the DEFAULT VLLM_CACHE_ROOT (/root/.cache/vllm), which
# still contains the poisoned pre-patch artifact torch_aot_compile/63eba2e...
# so that patch 002's cache-key invalidation is actually exercised.
#
# Own PGID (setsid), own port, own log => stoppable by PID without touching
# the coordinator's processes.
set -u
ME=/home/qiba/ROCm.AI/hyperloom/session/Ornith-1.5-397B-FP8/20260915T161838Z-ba2694ea/runs/specialist/8628519056b24176b3fa5c9a0e106dd6
MODEL=/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-FP8
PORT=${PORT:-8891}

echo "pgid=$(ps -o pgid= -p $$ | tr -d ' ') pid=$$ started=$(date -u +%FT%TZ)" > "$ME/verify.pid"

export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$ME/wt"
export TORCHINDUCTOR_CACHE_DIR="$ME/vcache/inductor"
export VLLM_ROCM_USE_AITER=0
export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_LOGGING_LEVEL=INFO
export PYTHONDONTWRITEBYTECODE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200

cd "$ME"
exec /opt/envs/vllm/bin/python /usr/local/bin/vllm serve "$MODEL" \
  --port "$PORT" --tensor-parallel-size=8 --gpu-memory-utilization "${GMU:-0.95}" \
  --max-model-len 6144 --trust-remote-code ${EXTRA:-}
