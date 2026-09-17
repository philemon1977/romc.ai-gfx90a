#!/usr/bin/env bash
# MoE tile-table A/B for MI250X (the one MoE lever that needs no kernel math).
# Each arm = one server start (~5.5 min) + 5 reps of the fixed count prompt with the
# first request discarded => 0.1% precision, so even a ~1% effect is judgeable.
#
# The experiment doubles as a MEASUREMENT OF THE MoE SHARE: if a 2x-better tile gives
# +3% end-to-end, the MoE owns ~17% of the step; if it gives +1%, ~6%.
set -uo pipefail
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
REPO=/home/qiba/ROCm.AI/quark-int8
TUNED="$REPO/moe_tuned"
NAME='E=512,N=128,device_name=AMD_Instinct_MI250X_MI250,dtype=int4_w4a16.json'
LOG="$REPO/logs/moe_table_ab.log"; mkdir -p "$REPO/logs" "$TUNED"
exec > >(tee -a "$LOG") 2>&1
echo "=== moe_table_ab start $(date -u +%FT%TZ) ==="

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

run_arm () {  # tag  "BM BN BK GROUP warps stages" | "default"
  local tag="$1" spec="$2"
  local pidf=/home/qiba/ai/logs/ornith397b-8116.pid
  [ -f "$pidf" ] && { local p; p=$(cat "$pidf" 2>/dev/null || true); [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null; rm -f "$pidf"; }
  wait_free || { echo "[$tag] GPU 不空"; return 1; }
  export PORT=8116 SPEC=5
  if [ "$spec" = "default" ]; then
    unset VLLM_TUNED_CONFIG_FOLDER
    rm -f "$TUNED/$NAME"
    echo; echo "=== ARM $tag  (无调优表 = vLLM 默认启发式) $(date -u +%H:%M:%S) ==="
  else
    python3 "$REPO/make_moe_table.py" "$TUNED/$NAME" $spec >/dev/null || return 1
    export VLLM_TUNED_CONFIG_FOLDER="$TUNED"
    echo; echo "=== ARM $tag  tiles: $spec $(date -u +%H:%M:%S) ==="
  fi
  bash "$LAUNCH" >/dev/null || { echo "[$tag] LAUNCH FAILED"; return 1; }
  for i in $(seq 1 200); do curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && break; sleep 5; done
  if ! curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1; then
    echo "[$tag] ❌ NOT READY"; local srv; srv=$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log | head -1)
    grep -aoE "(ValueError|RuntimeError|AssertionError|KeyError): .{0,140}" "$srv" | sort -u | head -3; return 1
  fi
  local srv; srv=$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log | head -1)
  echo "  --- MoE config 来源（必须出现 Using configuration from …）---"
  grep -aoE "Using configuration from [^\"]{0,110}|Using default MoE config" "$srv" | sort -u | head -2 | sed 's/^/    /'
  python3 "$REPO/measure_median.py" 8116 "$tag" 5 256 count
  local p2; p2=$(cat "$pidf" 2>/dev/null || true); [ -n "${p2:-}" ] && kill -TERM -"$p2" 2>/dev/null
  sleep 20
}

# batch2：小 tile 方向（默认 16/64/32 的"更小"一侧；vLLM 在 M=1 时用 16/32/64）
run_arm tbl-n32k64  "16 32 64 1 4 2"
run_arm tbl-n32k32  "16 32 32 1 2 2"
run_arm tbl-k64     "16 64 64 1 4 2"
run_arm tbl-k16     "16 64 16 1 4 2"
# 复核 batch1 的默认臂（同会话基线）
run_arm tbl-default "16 64 32 1 4 2"
echo "=== moe_table_ab done $(date -u +%FT%TZ) ==="
