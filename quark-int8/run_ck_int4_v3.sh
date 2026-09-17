#!/usr/bin/env bash
# CK int4 W4A16 GEMM numeric verification on gfx90a — CORRECTED (2nd fix).
# common.hpp requires argc>=10 for the SplitK variant: verify init time M N K StrideA StrideB StrideC [KBatch]
# (M/N/K are LITERAL values; defaults 3840/4096/4096). My first two attempts passed 6 args / used -v.
set -uo pipefail
OUT=/home/qiba/ROCm.AI/quark-int8/ck_int4
LOG=/home/qiba/ROCm.AI/quark-int8/logs/ck_int4_run3.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== run_ck_int4_v3 start $(date -u +%FT%TZ) ==="

for _ in $(seq 1 240); do
  busy=$(docker ps --format '{{.Names}}' | grep -c '^ornith-' || true)
  nc -z 127.0.0.1 8116 2>/dev/null && { sleep 15; continue; }
  nc -z 127.0.0.1 8115 2>/dev/null && { sleep 15; continue; }
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "$busy" = "0" ] && [ "${free:-0}" -ge 61 ] && break
  sleep 15
done
echo "GPU 空闲（最空 ${free:-?} GiB）"

run_one () {  # label verify init time M N K KBatch
  local label=$1 v=$2 i=$3 t=$4 m=$5 n=$6 k=$7 kb=$8
  echo
  echo "################ $label"
  ( cd "$OUT" && timeout 1200 env LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib ./"$BIN" "$v" "$i" "$t" "$m" "$n" "$k" -1 -1 -1 "$kb" ) 2>&1
  echo "######## 退出码=$?"
}

for BIN in gemm_xdl_bf16_pk_i4_v3.gfx90a gemm_xdl_fp16_pk_i4_v3_b_scale.gfx90a; do
  [ -x "$OUT/$BIN" ] || { echo "❌ 缺 $OUT/$BIN"; continue; }
  echo
  echo "==================== $BIN ===================="
  # 1) 小尺寸、CPU 参考对拍（只要数值正确性）
  run_one "M=256 N=2048 K=4096, verify=CPU, no timing" 1 2 0 256 2048 4096 1
  # 2) 默认尺寸（3840x4096x4096）CPU+GPU 双向对拍 + 计时
  run_one "默认 3840x4096x4096, verify=CPU+GPU, timing=yes" 3 2 1 3840 4096 4096 1
  # 3) 接近真实 MoE 分片（大 M 计时）
  run_one "M=2048 N=2048 K=8192, verify=CPU, timing=yes" 1 2 1 2048 2048 8192 1
done
echo "=== run_ck_int4_v3 done $(date -u +%FT%TZ) ==="
