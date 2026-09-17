#!/usr/bin/env bash
# C step: 640K context via YaRN 2.5x on the INT8 checkpoint, MTP on, zero offload.
set -uo pipefail
NAME=ornith-c640k
MAXLEN=${MAXLEN:-640000}
UTIL=${UTIL:-0.975}
OVERRIDES='{"max_position_embeddings":640000,"text_config":{"max_position_embeddings":640000,"rope_parameters":{"rope_type":"yarn","factor":2.45,"original_max_position_embeddings":262144,"rope_theta":10000000,"partial_fraction":0.25,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}}}'
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
  --served-model-name Ornith-640k --port 8100 --tensor-parallel-size 8 \
  --gpu-memory-utilization "$UTIL" --max-model-len "$MAXLEN" \
  --max-num-batched-tokens 2048 --max-num-seqs 2 \
  --hf-overrides "$OVERRIDES" \
  --language-model-only --trust-remote-code --moe-backend triton \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' >/dev/null
echo "launched $NAME (maxlen=$MAXLEN util=$UTIL YaRN 2.45x)"
