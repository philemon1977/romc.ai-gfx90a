#!/usr/bin/env bash
# CK int4 W4A16 GEMM numeric verification on gfx90a — CORRECTED invocation.
# The examples take POSITIONAL args (my first attempt wrongly passed `-v 1`, which just
# printed usage): arg1=verify(0/1=CPU ref/2=GPU/3=both) arg2=init(1=int,2=decimal)
# arg3=time arg4..9 = M(256x) N(128x) K(32x) strides arg10=KBatch
set -uo pipefail
OUT=/home/qiba/ROCm.AI/quark-int8/ck_int4
LOG=/home/qiba/ROCm.AI/quark-int8/logs/ck_int4_run2.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== run_ck_int4_v2 start $(date -u +%FT%TZ) ==="

for _ in $(seq 1 240); do
  busy=$(docker ps --format '{{.Names}}' | grep -c '^ornith-' || true)
  nc -z 127.0.0.1 8116 2>/dev/null && { sleep 15; continue; }
  nc -z 127.0.0.1 8115 2>/dev/null && { sleep 15; continue; }
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "$busy" = "0" ] && [ "${free:-0}" -ge 61 ] && break
  sleep 15
done
echo "GPU 空闲（最空 ${free:-?} GiB）"

run_one () {  # bin verify init time M256x N128x K32x label
  local bin=$1 v=$2 i=$3 t=$4 m=$5 n=$6 k=$7 label=$8
  echo
  echo "=== $(basename "$bin")  [$label]  M=$((m*256)) N=$((n*128)) K=$((k*32)) ==="
  ( cd "$OUT" && timeout 900 env LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib "./$bin" "$v" "$i" "$t" "$m" "$n" "$k" ) 2>&1 \
    | grep -aiE "verify|error|pass|fail|max|check|Mismatch|launch|time|GB/s|TFlops|^\[" | head -18
  echo "  (退出码 ${PIPESTATUS[0]:-?})"
}

for b in gemm_xdl_bf16_pk_i4_v3.gfx90a gemm_xdl_fp16_pk_i4_v3_b_scale.gfx90a; do
  [ -x "$OUT/$b" ] || { echo "❌ 缺 $OUT/$b"; continue; }
  # CPU 参考对拍，小尺寸先确认数值；再上接近单专家形状的大尺寸
  run_one "$b" 1 2 0 1 16 128 "M=256 N=2048 K=4096（≈单专家 gate/up 分片）"
  run_one "$b" 1 2 1 8 16 256 "M=2048 N=2048 K=8192（大 M，计时）"
done
echo "=== run_ck_int4_v2 done $(date -u +%FT%TZ) ==="
