#!/usr/bin/env bash
# Run the gfx90a-built CK int4 (pk_i4) W4A16 GEMM examples once the GPU is free.
# A PASS here means: CK's packed-int4 W4A16 GEMM is *numerically correct on gfx90a*,
# i.e. the earlier "no int4 kernel exists for CDNA2" verdict was wrong in its premise.
# Deliberately serialized after the MTP sweep: a bad int4 device path can hang the GCD.
set -uo pipefail
OUT=/home/qiba/ROCm.AI/quark-int8/ck_int4
LOG=/home/qiba/ROCm.AI/quark-int8/logs/ck_int4_run.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== run_ck_int4 start $(date -u +%FT%TZ) ==="

for _ in $(seq 1 240); do
  busy=$(docker ps --format '{{.Names}}' | grep -c '^ornith-' || true)
  nc -z 127.0.0.1 8115 2>/dev/null && { sleep 15; continue; }
  nc -z 127.0.0.1 8116 2>/dev/null && { sleep 15; continue; }
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "$busy" = "0" ] && [ "${free:-0}" -ge 61 ] && break
  sleep 15
done
echo "GPU 空闲（最空 die ${free:-?} GiB）——开始跑 int4 kernel"

for b in gemm_xdl_bf16_pk_i4_v3 gemm_xdl_fp16_pk_i4_v3_b_scale; do
  [ -x "$OUT/$b.gfx90a" ] || { echo "❌ 缺产物 $OUT/$b.gfx90a"; continue; }
  echo
  echo "=== $b ==="
  echo "--- 默认尺寸下的数值校验（ReferenceGemm 对拍）---"
  timeout 600 env LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib "$OUT/$b.gfx90a" -v 1 2>&1 | tail -20
  echo "退出码=$?"
  echo "--- 较大尺寸 M=2048 N=2048 K=8192（更接近真实 MoE 分片形状）---"
  timeout 900 env LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib "$OUT/$b.gfx90a" -v 1 -M 2048 -N 2048 -K 8192 2>&1 | tail -20
  echo "退出码=$?"
done
echo "=== run_ck_int4 done $(date -u +%FT%TZ) ==="
