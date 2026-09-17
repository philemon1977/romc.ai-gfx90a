#!/usr/bin/env bash
# Clean single-variable A/B: AITER int8 Linear ON vs OFF, on the INT8-v2 (Quark
# Int8-Attn) 256K + MTP(1) recipe — i.e. exactly the arm that competes with int4.
#
#   T (control): VLLM_ROCM_USE_AITER=0            -> expect Selected TritonInt8ScaledMMLinearKernel
#   A (aiter)  : VLLM_ROCM_USE_AITER=1 + patch    -> expect Selected AiterInt8ScaledMMLinearKernel
#
# The master switch is DANGEROUS on gfx90a: platforms/rocm.py:1109-1116 gates RMSNorm on
# VLLM_ROCM_USE_AITER [+RMSNORM sub-switch] with NO arch term, while module_norm.so has
# zero gfx90a code objects. So every other aiter sub-switch is forced to 0 here.
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int8w8a8_vllm_rocm72_mtp1_256k_8115_ornith_mi250dx8.sh
LOG="$REPO/logs/ab_int8.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== ab_int8 start $(date -u +%FT%TZ) ==="

wait_free () {
  for _ in $(seq 1 120); do
    busy=$(docker ps --format '{{.Names}}' | grep -c '^ornith-' || true)
    nc -z 127.0.0.1 8115 2>/dev/null && { sleep 10; continue; }
    nc -z 127.0.0.1 8116 2>/dev/null && { sleep 10; continue; }
    free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    [ "$busy" = "0" ] && [ "${free:-0}" -ge 62 ] && return 0
    sleep 10
  done
  return 1
}

run_arm () {  # tag aiter_on
  local tag=$1 on=$2
  local pidf=/home/qiba/ai/logs/ornith397b-8115.pid
  # stop whatever is on 8115
  if [ -f "$pidf" ]; then
    local p; p=$(cat "$pidf" 2>/dev/null || true)
    [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null && echo "[$tag] 已 TERM 上一实例 $p"
  fi
  rm -f "$pidf"
  wait_free || { echo "[$tag] GPU 一直不空"; return 1; }

  export PORT=8115
  # ⚠️ 这里不要加 --load-format fastsafetensors：int8 权重 51.75 GiB/die（util 0.96，可用 61.4），
  # fastsafetensors 批量搬 shard 会一次申请 ~12.3 GiB 而只剩 8.9 GiB 空闲 ⇒ 加载期 OOM
  # （实测 2026-09-17 GPU2：Consumer error: CUDA out of memory / gpu_model_runner.py:5513）。
  # 该 format 只在本模型 int4（28.4 GiB/die）上安全。
  unset VLLM_EXTRA_ARGS
  if [ "$on" = "1" ]; then
    export PYTHONPATH="$REPO/aiter_patch"
    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_LINEAR=1
    export VLLM_ROCM_USE_AITER_RMSNORM=0        # ← gfx90a 雷，必须显式关
    export VLLM_ROCM_USE_AITER_MOE=0
    export VLLM_ROCM_USE_AITER_MLA=0
    export VLLM_ROCM_USE_AITER_MHA=0
    export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
    export VLLM_ROCM_USE_AITER_FP8BMM=0
    export VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=0
    export VLLM_ROCM_USE_AITER_TRITON_ROPE=0
    export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
    export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=0
    export VLLM_ROCM_USE_AITER_TRITON_GEMM=0
  else
    unset PYTHONPATH VLLM_ROCM_USE_AITER_LINEAR VLLM_ROCM_USE_AITER_RMSNORM
    export VLLM_ROCM_USE_AITER=0
  fi
  echo "=== ARM $tag  AITER=$VLLM_ROCM_USE_AITER  PYTHONPATH=${PYTHONPATH:-（无）} $(date -u +%H:%M:%S) ==="
  bash "$LAUNCH" || { echo "[$tag] LAUNCH FAILED"; return 1; }

  for i in $(seq 1 200); do
    curl -sf http://127.0.0.1:8115/health >/dev/null 2>&1 && { echo "[$tag] ✅ ready (~$((i*5))s)"; break; }
    sleep 5
  done
  local srv; srv=$(ls -t /home/qiba/ai/logs/ornith397b/server-8115-*.log | head -1)
  if ! curl -sf http://127.0.0.1:8115/health >/dev/null 2>&1; then
    echo "[$tag] ❌ NOT READY："
    grep -aoE "(ValueError|RuntimeError|AssertionError|AttributeError|ImportError): .{0,170}" "$srv" | sort -u | head -5
    return 1
  fi
  echo "--- 关键行（$srv）---"
  grep -aoE "Selected [A-Za-z0-9]*Int8ScaledMMLinearKernel|\[aiter\] import \[module_gemm_a8w8\]|\[gfx90a-patch\][^\"]{0,60}|Loading weights took [0-9.]+ seconds|GPU KV cache size: [0-9,]+ tokens.*|runtime shape|Using [A-Za-z_']+ MoE backend" "$srv" | sort | uniq -c | sort -rn | head -8

  echo "--- 单流 TPS（256 token, greedy, MTP）---"
  python3 - <<PY
import json,time,urllib.request
body=json.dumps({'model':'ornith','prompt':'Explain in detail why int4 quantization reduces memory bandwidth pressure during decoding.','max_tokens':256,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8115/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=1800)); dt=time.time()-t0
u=d['usage']; print('[$tag] SINGLE-STREAM %.2f tok/s (%d tok in %.1fs)'%(u['completion_tokens']/dt,u['completion_tokens'],dt))
PY
  curl -s localhost:8115/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{" | sed 's/^/  /'
  local p2; p2=$(cat /home/qiba/ai/logs/ornith397b-8115.pid 2>/dev/null || true)
  [ -n "${p2:-}" ] && kill -TERM -"$p2" 2>/dev/null && echo "[$tag] 已停服 $p2"
  sleep 20
}

run_arm int8-256k-triton 0
run_arm int8-256k-aiter  1
echo "=== ab_int8 done $(date -u +%FT%TZ) ==="
