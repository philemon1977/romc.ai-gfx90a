#!/bin/bash
cd /home/qiba/ROCm.AI/hyperloom/session/Ornith-1.5-397B-FP8/20260915T161838Z-ba2694ea/runs/specialist/fc07edbde4c9489682f2cafa5a7d253a
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/home/qiba/ROCm.AI/hyperloom/session/Ornith-1.5-397B-FP8/20260915T161838Z-ba2694ea/runs/specialist/fc07edbde4c9489682f2cafa5a7d253a/worktree
export VLLM_CACHE_ROOT=/home/qiba/ROCm.AI/hyperloom/session/Ornith-1.5-397B-FP8/20260915T161838Z-ba2694ea/runs/specialist/fc07edbde4c9489682f2cafa5a7d253a/vcache
export TORCHINDUCTOR_CACHE_DIR=/home/qiba/ROCm.AI/hyperloom/session/Ornith-1.5-397B-FP8/20260915T161838Z-ba2694ea/runs/specialist/fc07edbde4c9489682f2cafa5a7d253a/vcache/inductor
export VLLM_ROCM_USE_AITER=0
export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_LOGGING_LEVEL=INFO
exec /usr/local/bin/vllm serve /mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-FP8   --port 8889 --tensor-parallel-size=8 --gpu-memory-utilization 0.95   --max-model-len 6144 --trust-remote-code
