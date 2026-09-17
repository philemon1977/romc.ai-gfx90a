#!/usr/bin/env bash
# (C) Measure the NATIVE fp4 flatmm MoE (ck_tile, already verified correct on gfx90a) at the
# REAL shapes of this model, so its cost can be compared against the measured step time
# (48-63 ms/step at SPEC=5). Single-sided measurement: no need to also microbenchmark the
# Triton wna16 kernel — if 60 x (CK MoE per layer) is a small slice of the step, then even a
# perfect MoE kernel cannot pay for the wiring effort.
#
# Model shapes: hidden=4096, moe_intermediate_size=1024, 512 experts, topk=10.
#   gemm1_gate_up : N = 2*moe_inter = 2048, K = hidden = 4096
#   gemm2         : N = hidden = 4096,      K = moe_inter = 1024
set -uo pipefail
OUT=/home/qiba/ROCm.AI/quark-int8/ck_int4
LOG=/home/qiba/ROCm.AI/quark-int8/logs/fp4_moe_bench.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== fp4_moe_bench start $(date -u +%FT%TZ) ==="

# 停掉可能还在跑的服务（benchmark 需要独占 GCD 的显存）
pidf=/home/qiba/ai/logs/ornith397b-8116.pid
[ -f "$pidf" ] && { p=$(cat "$pidf" 2>/dev/null || true); [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null && echo "已停服 $p"; rm -f "$pidf"; }
for _ in $(seq 1 120); do
  nc -z 127.0.0.1 8116 2>/dev/null && { sleep 10; continue; }
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "${free:-0}" -ge 60 ] && break
  sleep 10
done
echo "GPU 最空 ${free:-?} GiB"

BIN="$OUT/a16w4_moe_flatmm.gfx90a"
[ -x "$BIN" ] || { echo "❌ 缺 $BIN"; exit 1; }

run_case () {  # kind tokens N K label
  local kind=$1 tok=$2 N=$3 K=$4 label=$5
  echo
  echo "################ $label  gemm_kind=$kind NumTokens=$tok N=$N K=$K experts=512 topk=10"
  ( cd "$OUT" && timeout 1800 env LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib ./a16w4_moe_flatmm.gfx90a \
      -gemm_kind="$kind" -mixed_prec=bf16xfp4 \
      -experts=512 -TopK=10 -NumTokens="$tok" -N="$N" -K="$K" \
      -validate=0 -init=0 -warmup=30 -repeat=60 -k_batch=1 ) 2>&1 \
    | grep -aiE "Launching kernel|Perf:|error|terminate|abort|invalid" | head -6
}

echo "=== gemm1_gate_up（N=2048=2×moe_inter, K=4096=hidden）==="
for tok in 16 64 256; do run_case gemm1_gate_up "$tok" 2048 4096 "gemm1 tokens=$tok"; done
echo
echo "=== gemm2（N=4096=hidden, K=1024=moe_inter）==="
for tok in 16 64 256; do run_case gemm2 "$tok" 4096 1024 "gemm2 tokens=$tok"; done
echo "=== fp4_moe_bench done $(date -u +%FT%TZ) ==="
