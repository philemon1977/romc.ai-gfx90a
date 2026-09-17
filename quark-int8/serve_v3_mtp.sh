#!/usr/bin/env bash
# Serve Ornith-1.5-397B MXFP4 (W4A16, Quark OCP-MX) with MTP speculative decoding
# on 8x MI250X (gfx90a).  MXFP4 MoE runs through vLLM's emulation backend
# (dequant + bf16 GEMM): memory-first configuration, not a compute win.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-Quark-MXFP4-W4A16}
NAME=${NAME:-ornith-mxfp4-mtp}
PORT=${PORT:-8100}
MAX_LEN=${MAX_LEN:-32768}
MEM_UTIL=${MEM_UTIL:-0.90}
MAX_SEQS=${MAX_SEQS:-16}
SPEC_TOKENS=${SPEC_TOKENS:-1}

docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --network host \
  --device /dev/kfd --device /dev/dri --group-add video \
  --shm-size 64G --ulimit memlock=-1:-1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=0 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 \
  vllm/vllm-openai-rocm:nightly \
    "$MODEL_DIR" \
    --served-model-name Ornith-1.5-397B-MXFP4-W4A16 \
    --port "$PORT" \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization "$MEM_UTIL" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --max-num-batched-tokens 2048 \
    --language-model-only \
    --trust-remote-code \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}"

echo "serving container: $NAME (port $PORT, MTP=${SPEC_TOKENS} tokens)"
echo "watch:  ./watch_container.sh $NAME 1800 $PORT"
echo "evidence: PORT=$PORT MODEL=Ornith-1.5-397B-MXFP4-W4A16 ./evidence_v3_mtp.sh"
