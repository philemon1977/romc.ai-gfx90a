#!/usr/bin/env bash
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
NAME=ornith-astep
docker rm -f "$NAME" >/dev/null 2>&1 || true
for i in $(seq 1 60); do
  f=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "${f:-0}" -ge 62 ] && break; sleep 10
done
docker run -d --name "$NAME" --network host \
  --device /dev/kfd --device /dev/dri --group-add video --shm-size 64G --ulimit memlock=-1:-1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 vllm/vllm-openai-rocm:nightly \
  /mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-Quark-Int8-Attn \
  --served-model-name Ornith-astep --port 8100 --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.975 --max-model-len 262144 \
  --max-num-batched-tokens 2048 --max-num-seqs 2 \
  --language-model-only --trust-remote-code --moe-backend triton \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' >/dev/null
echo "launched $NAME (util 0.975, seqs 2, graph mode, no cache)"
