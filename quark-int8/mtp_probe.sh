#!/usr/bin/env bash
# Task ②: measure the real VRAM cost of MTP + long context, and whether they fit.
# Usage: mtp_probe.sh <tag> <max_model_len> <max_num_seqs> <util> [mtp_tokens]
#   tag=base      -> no speculative decoding
#   tag=mtp       -> --speculative-config '{"method":"mtp","num_speculative_tokens":N}'
set -uo pipefail
TAG=${1:?tag}
MAXLEN=${2:-131072}
SEQS=${3:-16}
UTIL=${4:-0.96}
MTPN=${5:-1}
MODEL_DIR=${MODEL_DIR:-/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-Quark-Int8-Attn}
NAME=ornith-${TAG}
LOG_DIR=/home/qiba/ROCm.AI/quark-int8/logs
mkdir -p "$LOG_DIR"

SPEC=()
if [ "$TAG" = "mtp" ]; then
  SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTPN}}")
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
# HBM is not released instantly when another container dies -> wait for the full
# budget on every GCD, otherwise vLLM aborts with "Free memory ... is less than
# desired GPU memory utilization".
NEED_GB=$(python3 -c "print(int(${UTIL}*63.98))")
echo "waiting for >= ${NEED_GB} GiB free on every GCD..."
for i in $(seq 1 60); do
  minfree=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 \
            | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  if [ "${minfree:-0}" -ge "$NEED_GB" ]; then echo "  ok (min free ${minfree} GiB)"; break; fi
  sleep 10
done
docker run -d --name "$NAME" --network host \
  --device /dev/kfd --device /dev/dri --group-add video --shm-size 64G --ulimit memlock=-1:-1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 vllm/vllm-openai-rocm:nightly \
  "$MODEL_DIR" \
  --served-model-name "Ornith-$TAG" --port 8100 --tensor-parallel-size 8 \
  --gpu-memory-utilization "$UTIL" --max-model-len "$MAXLEN" \
  --max-num-batched-tokens 2048 --max-num-seqs "$SEQS" \
  --language-model-only --trust-remote-code \
  --moe-backend "${MOE_BACKEND:-triton}" "${SPEC[@]}" >/dev/null

echo "launched $NAME (maxlen=$MAXLEN seqs=$SEQS util=$UTIL tag=$TAG)"
docker logs -f "$NAME" > "$LOG_DIR/${TAG}_$(date +%H%M).log" 2>&1 &
echo "log: $LOG_DIR/${TAG}_*.log"
