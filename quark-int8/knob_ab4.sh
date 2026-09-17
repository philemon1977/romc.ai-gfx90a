#!/usr/bin/env bash
# Batch 4 — last shots in config space, all with the adopted baseline
# (SPEC=5 + --no-enable-prefix-caching) and the 0.1%-precision methodology.
#
#   ctrl+stats : control + VLLM_ROCM_SPLITKV_PA_STATS -> does the project's split-KV PA
#                patch EVER take over under MTP? (its gate needs max_query_len==1, and the
#                launcher's echo line claims "SPEC>0 时不接管" — this verifies that claim
#                instead of assuming it, at zero extra cost)
#   async      : --async-scheduling (overlap CPU scheduling with GPU execution)
#   bs128/bs256: --block-size; the effective attention block size is currently auto-set to
#                ~544 tokens by the mamba page-size alignment, so overriding it is untested
set -uo pipefail
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
REPO=/home/qiba/ROCm.AI/quark-int8
LOG="$REPO/logs/knob_ab4.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== knob_ab4 start $(date -u +%FT%TZ) ==="

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

run_arm () {  # tag extra_args extra_env("VAR=V ...")
  local tag="$1" extra="${2:-}" eenv="${3:-}"
  local pidf=/home/qiba/ai/logs/ornith397b-8116.pid
  [ -f "$pidf" ] && { local p; p=$(cat "$pidf" 2>/dev/null || true); [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null; rm -f "$pidf"; }
  wait_free || { echo "[$tag] GPU 不空"; return 1; }
  export PORT=8116 SPEC=5
  unset VLLM_EXTRA_ARGS VLLM_ROCM_SPLITKV_PA_STATS
  [ -n "$extra" ] && export VLLM_EXTRA_ARGS="$extra"
  local kv; for kv in $eenv; do export "$kv"; done
  echo
  echo "=== ARM $tag EXTRA='${VLLM_EXTRA_ARGS:-}' ENV='${eenv:-}' $(date -u +%H:%M:%S) ==="
  bash "$LAUNCH" >/dev/null || { echo "[$tag] LAUNCH FAILED"; return 1; }
  for i in $(seq 1 200); do curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && break; sleep 5; done
  if ! curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1; then
    echo "[$tag] ❌ NOT READY"; local srv; srv=$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log | head -1)
    grep -aoE "(ValueError|RuntimeError|AssertionError|TypeError): .{0,140}" "$srv" | sort -u | head -3; return 1
  fi
  local srv; srv=$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log | head -1)
  grep -aoE "Setting attention block size to [0-9]+ tokens[^\"]{0,50}|Mamba cache mode is set to '[a-z]+'|GPU KV cache size: [0-9,]+ tokens.*|Async scheduling[^\"]{0,40}" "$srv" | sort -u | head -4 | sed 's/^/    /'
  python3 "$REPO/measure_median.py" 8116 "$tag" 5 256 count
  if [ -n "${VLLM_ROCM_SPLITKV_PA_STATS:-}" ]; then
    echo "  --- splitKV 接管统计 ---"; cat "$VLLM_ROCM_SPLITKV_PA_STATS" 2>/dev/null | head -20 || echo "    （统计文件为空/不存在 ⇒ 从未接管）"
  fi
  local p2; p2=$(cat "$pidf" 2>/dev/null || true); [ -n "${p2:-}" ] && kill -TERM -"$p2" 2>/dev/null
  sleep 20
}

run_arm ctrl-stats "" "VLLM_ROCM_SPLITKV_PA_STATS=$REPO/logs/splitkv_stats.json"
run_arm async "--async-scheduling"
run_arm bs128 "--block-size 128"
run_arm bs256 "--block-size 256"
echo "=== knob_ab4 done $(date -u +%FT%TZ) ==="
