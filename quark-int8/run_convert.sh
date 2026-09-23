#!/usr/bin/env bash
# Convert DSV4.1-Flash -> compressed-tensors INT4 W4A16 (vLLM W4A16 kernels on gfx90a).
#
# HOST_OUT is the host-side output dir; it is mapped into the container as /out
# (never pass the host path to the converter itself).
#
#   ./run_convert.sh                        # pilot: shard 3 -> $REPO/pilot_ct_int4
#   SHARDS=3,4 ./run_convert.sh             # specific shards
#   SHARDS= SKIP_EXISTING=1 ./run_convert.sh   # full run, resuming what's done
#   MIN_FREE_GIB=2 ALLOW_CPU=1 ...          # fall back to CPU if GPUs are busy
set -uo pipefail

REPO=/home/qiba/ROCm.AI/quark-int8
SRC=${SRC:-/mnt/stripe-3mix-3t2/models/deepseek-ai/DeepSeek-V4.1-Flash}
HOST_OUT=${HOST_OUT:-$REPO/pilot_ct_int4}
SHARDS=${SHARDS-3}
GPU=${GPU-}

EXTRA=()
[ -n "${SKIP_EXISTING:-}" ] && EXTRA+=(--skip-existing)
[ -n "${ALLOW_CPU:-}" ] && EXTRA+=(--allow-cpu)
# 逐组最优 int4 scale：实测把 int4 相对误差 10.0%→6.8%（fp4 专家最好 1.6×，见转换记录 §4.16）。
# 步长默认 0.05（6 点）：实测相对 15 点仅变差 +1.887%，但这段 CPU 耗时降到 0.46×（§4.20）。
[ -n "${ATTN_MERGED:-}" ] && EXTRA+=(--attn-merged-bf16)
[ -n "${ATTN_BF16:-}" ] && EXTRA+=(--attn-bf16)
[ -n "${SHARED_BF16:-}" ] && EXTRA+=(--shared-experts-bf16)
[ -n "${OPT_SCALE:-}" ] && EXTRA+=(--optimal-scale
  --opt-scale-lo "${OPT_LO:-0.72}" --opt-scale-hi "${OPT_HI:-1.00}" --opt-scale-step "${OPT_STEP:-0.05}")

mkdir -p "$HOST_OUT"

CMD="python3 /w/quark-int8/convert_dsv41_ct_int4.py --model $SRC --out /out"
CMD="$CMD --device ${DEVICE:-cuda} --min-free-gib ${MIN_FREE_GIB:-2}"
CMD="$CMD --wait-max-s ${WAIT_MAX_S:-1800} --wait-poll-s ${WAIT_POLL_S:-15}"
[ -n "$SHARDS" ] && CMD="$CMD --shards $SHARDS"
[ ${#EXTRA[@]} -gt 0 ] && CMD="$CMD ${EXTRA[*]}"

echo "[run] host_out=$HOST_OUT  shards=${SHARDS:-ALL}  gpu=${GPU:-ALL}"

DOCKER_ARGS=(
  --rm --entrypoint bash
  --device /dev/kfd --device /dev/dri
  --group-add 44 --group-add 993 --user 1000:1000
  -e HOME=/tmp -e USER=qiba
  -v /home/qiba/ROCm.AI:/w
  -v /sys/class/drm:/sys/class/drm:ro
  -v "$SRC":"$SRC":ro
  -v "$HOST_OUT":/out
  --workdir /w/quark-int8
)
[ -n "$GPU" ] && DOCKER_ARGS+=(-e HIP_VISIBLE_DEVICES="$GPU")

exec docker run "${DOCKER_ARGS[@]}" vllm/vllm-openai-rocm:nightly -c "$CMD"
