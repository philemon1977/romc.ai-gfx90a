#!/usr/bin/env bash
# Expert-cache V1 probe. Usage: ec_probe.sh <tag> <offload_pct> <eager 0|1> [util]
# Example:  ./ec_probe.sh ec12 12 1          # 12% cold experts, eager, cache ON
#           ./ec_probe.sh base0 0 1          # matched baseline, eager, cache OFF
set -uo pipefail
TAG=${1:?tag}
PCT=${2:-12}
EAGER=${3:-1}
UTIL=${4:-0.96}
NAME=ornith-$TAG
REPO=/home/qiba/ROCm.AI/quark-int8
MODEL_DIR=${MODEL_DIR:-/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-Quark-Int8-Attn}

EC_ENV=()
EC_MOUNT=()
if [ "$PCT" != "0" ]; then
  EC_ENV=(-e EXPERT_CACHE=1 -e EXPERT_CACHE_PCT="$PCT"
          -e EXPERT_CACHE_MODE="${EC_MODE:-sim}"
          -e EXPERT_CACHE_POOL="${EC_POOL:-8}"
          -e EXPERT_CACHE_HOTLIST=/ec/hotlist.npz
          -e EXPERT_CACHE_STATS=/work/expert_cache_stats.json
          -e PYTHONPATH=/ec)
  EC_MOUNT=(-v "$REPO/expert_cache:/ec")
fi
EAGER_ARG=()
[ "$EAGER" = "1" ] && EAGER_ARG=(--enforce-eager)

docker rm -f "$NAME" >/dev/null 2>&1 || true
NEED_GB=$(python3 -c "print(int(${UTIL}*63.98))")
for i in $(seq 1 60); do
  minfree=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 \
            | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "${minfree:-0}" -ge "$NEED_GB" ] && break
  sleep 10
done

docker run -d --name "$NAME" --network host \
  --device /dev/kfd --device /dev/dri --group-add video --shm-size 64G --ulimit memlock=-1:-1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  "${EC_ENV[@]}" \
  -v "$REPO:/work" "${EC_MOUNT[@]}" \
  -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 vllm/vllm-openai-rocm:nightly \
  "$MODEL_DIR" \
  --served-model-name "Ornith-$TAG" --port 8100 --tensor-parallel-size 8 \
  --gpu-memory-utilization "$UTIL" --max-model-len ${MAXLEN:-262144} \
  --max-num-batched-tokens 2048 --max-num-seqs 16 \
  --language-model-only --trust-remote-code --moe-backend triton \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  "${EAGER_ARG[@]}" >/dev/null

echo "launched $NAME (pct=$PCT eager=$EAGER util=$UTIL)"
