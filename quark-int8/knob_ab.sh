#!/usr/bin/env bash
# Single-variable ablations against the 27 ms/step fixed overhead, all at SPEC=5,
# each with a 5-rep median (single-shot noise is 12-24% here).
#
# Deliberately NOT tested (already adjudicated by the project):
#   * re-enabling custom all-reduce  -> CUSTOM silently computes wrong on gfx90a
#   * QUICK_REDUCE at TP8            -> measured and rejected (docs §4quater)
#   * NCCL_SYMM_MEM                  -> slower than plain RCCL here
#   * compilation level / cudagraph  -> already maximal (VLLM_COMPILE=3, FULL_AND_PIECEWISE)
set -uo pipefail
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
REPO=/home/qiba/ROCm.AI/quark-int8
LOG="$REPO/logs/knob_ab.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== knob_ab start $(date -u +%FT%TZ) ==="

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

run_arm () {  # tag extra_args "NCCL_VAR=V ..."
  local tag="$1" extra="${2:-}" ncl="${3:-}"
  local pidf=/home/qiba/ai/logs/ornith397b-8116.pid
  [ -f "$pidf" ] && { local p; p=$(cat "$pidf" 2>/dev/null || true); [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null; rm -f "$pidf"; }
  wait_free || { echo "[$tag] GPU 不空"; return 1; }

  export PORT=8116 SPEC=5
  unset VLLM_EXTRA_ARGS NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_PROTO NCCL_ALGO
  [ -n "$extra" ] && export VLLM_EXTRA_ARGS="$extra"
  local kv
  for kv in $ncl; do export "$kv"; done

  echo
  echo "=== ARM $tag  EXTRA='${VLLM_EXTRA_ARGS:-}'  NCCL='${ncl:-默认}' $(date -u +%H:%M:%S) ==="
  bash "$LAUNCH" >/dev/null || { echo "[$tag] LAUNCH FAILED"; return 1; }
  for i in $(seq 1 200); do curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && break; sleep 5; done
  if ! curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1; then
    echo "[$tag] ❌ NOT READY"; local srv; srv=$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log | head -1)
    grep -aoE "(ValueError|RuntimeError|AssertionError): .{0,150}" "$srv" | sort -u | head -3
    return 1
  fi
  local srv; srv=$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log | head -1)
  grep -aoE "cudagraph_mode': <CUDAGraphMode\.[A-Z_]+|fuse_allreduce_rms[^,]{0,20}|all-reduce backends \(in dispatch order\)[^\"]{0,60}" "$srv" | sort -u | head -3 | sed 's/^/    /'
  python3 "$REPO/measure_median.py" 8116 "$tag" 5 256
  local p2; p2=$(cat "$pidf" 2>/dev/null || true); [ -n "${p2:-}" ] && kill -TERM -"$p2" 2>/dev/null
  sleep 20
}

run_arm control ""
run_arm piecewise '--compilation-config {"cudagraph_mode":"PIECEWISE"}'
run_arm fuse-ar-rms '--compilation-config {"pass_config":{"fuse_allreduce_rms":true}}'
run_arm nccl-1chan "" "NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1"
echo "=== knob_ab done $(date -u +%FT%TZ) ==="
