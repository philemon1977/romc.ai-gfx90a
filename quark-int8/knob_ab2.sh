#!/usr/bin/env bash
# Batch 2 — re-run with the SENSITIVE methodology (fixed high-predictability prompt,
# 5 reps, first request excluded => spread ~1.5% instead of 12-35%).
# Batch 1's arms were unjudgeable inside the salted-prompt noise band; fuse_allreduce_rms
# is simply absent from this ROCm build ('AllReduceFusionPass' is not defined).
#
# Arms:
#   1) --no-enable-prefix-caching : config.py:602 forces mamba_cache_mode "none"->"align"
#      whenever prefix caching is on; "align" adds per-step page-alignment kernels
#      (precopy_mamba_align_fused_kernel / postprocess_mamba_fused_kernel), 3 padding
#      layers and wastes 6.67% KV. Under MTP the prefix cache never hits anyway
#      (measured: prefix_cache_queries 74,761 / hits 0) => expected pure win.
#   2) PIECEWISE cudagraph : retest with real SNR (batch 1 said -9.7% but inside noise).
#   3) --max-num-seqs 1 : minimal scheduler/graph surface for single stream.
set -uo pipefail
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
REPO=/home/qiba/ROCm.AI/quark-int8
LOG="$REPO/logs/knob_ab2.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== knob_ab2b start $(date -u +%FT%TZ) ==="

wait_free () {
  for _ in $(seq 1 180); do
    busy=$(docker ps --format '{{.Names}}' | grep -c '^ornith-' || true)
    nc -z 127.0.0.1 8116 2>/dev/null && { sleep 15; continue; }
    free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    [ "$busy" = "0" ] && [ "${free:-0}" -ge 62 ] && return 0
    sleep 15
  done
  return 1
}

run_arm () {  # tag extra_args
  local tag="$1" extra="${2:-}"
  local pidf=/home/qiba/ai/logs/ornith397b-8116.pid
  [ -f "$pidf" ] && { local p; p=$(cat "$pidf" 2>/dev/null || true); [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null; rm -f "$pidf"; }
  wait_free || { echo "[$tag] GPU 不空"; return 1; }
  export PORT=8116 SPEC=5
  unset VLLM_EXTRA_ARGS
  [ -n "$extra" ] && export VLLM_EXTRA_ARGS="$extra"
  echo
  echo "=== ARM $tag  EXTRA='${VLLM_EXTRA_ARGS:-}' $(date -u +%H:%M:%S) ==="
  bash "$LAUNCH" >/dev/null || { echo "[$tag] LAUNCH FAILED"; return 1; }
  for i in $(seq 1 200); do curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && break; sleep 5; done
  if ! curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1; then
    echo "[$tag] ❌ NOT READY"; local srv; srv=$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log | head -1)
    grep -aoE "(ValueError|RuntimeError|AssertionError): .{0,140}" "$srv" | sort -u | head -3; return 1
  fi
  local srv; srv=$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log | head -1)
  grep -aoE "Mamba cache mode is set to '[a-z]+'|Add [0-9]+ padding layers[^\"]{0,40}|GPU KV cache size: [0-9,]+ tokens.*" "$srv" | sort -u | head -3 | sed 's/^/    /'
  python3 "$REPO/measure_median.py" 8116 "$tag" 5 256 count
  local p2; p2=$(cat "$pidf" 2>/dev/null || true); [ -n "${p2:-}" ] && kill -TERM -"$p2" 2>/dev/null
  sleep 20
}

run_arm control-count ""
run_arm no-prefix-cache "--no-enable-prefix-caching"
run_arm piecewise '--compilation-config {"cudagraph_mode":"PIECEWISE"}'
run_arm seqs1 "--max-num-seqs 1"
echo "=== knob_ab2b done $(date -u +%FT%TZ) ==="
