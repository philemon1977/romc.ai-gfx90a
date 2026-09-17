#!/usr/bin/env bash
# Next batch, strictly serial (single-tenant GPUs):
#   1) int4 + MTP n=4        -> is the MTP peak past n=3? (pos2 still accepted 55% at n=3)
#   2) int4 + MTP n=5        -> locate the turn
#   3) int8 + AITER + MTP n=3-> combine the two measured wins (aiter linear +14.9%,
#                               MTP n=3 +33% on int4) and re-check quality (NLL) because the
#                               aiter arm's MTP acceptance dropped 86.1% -> 72.3% (numerics differ).
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH_INT4=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
LAUNCH_INT8=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int8w8a8_vllm_rocm72_mtp1_256k_8115_ornith_mi250dx8.sh
IMG=vllm/vllm-openai-rocm:nightly
LOG="$REPO/logs/next_batch.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== next_batch start $(date -u +%FT%TZ) ==="

wait_free () {
  for _ in $(seq 1 180); do
    busy=$(docker ps --format '{{.Names}}' | grep -c '^ornith-' || true)
    nc -z 127.0.0.1 8115 2>/dev/null && { sleep 15; continue; }
    nc -z 127.0.0.1 8116 2>/dev/null && { sleep 15; continue; }
    free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    [ "$busy" = "0" ] && [ "${free:-0}" -ge 62 ] && return 0
    sleep 15
  done
  return 1
}

stop_port () { # port
  local pidf=/home/qiba/ai/logs/ornith397b-$1.pid
  [ -f "$pidf" ] && { local p; p=$(cat "$pidf" 2>/dev/null || true); [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null; rm -f "$pidf"; }
}

measure () { # port tag
  local port=$1 tag=$2
  python3 - <<PY
import json,time,urllib.request
body=json.dumps({'model':'ornith','prompt':'Explain in detail why int4 quantization reduces memory bandwidth pressure during decoding.','max_tokens':256,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:$port/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=1800)); dt=time.time()-t0
u=d['usage']; print('[$tag] SINGLE-STREAM %.2f tok/s (%d tok in %.1fs)'%(u['completion_tokens']/dt,u['completion_tokens'],dt))
PY
  curl -s localhost:$port/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{|per_pos_total\{" | sed "s/^/  [$tag] /"
}

run_int4 () { # spec
  local n=$1; local tag="int4-mtp$n"
  stop_port 8116; wait_free || { echo "[$tag] GPU 不空"; return 1; }
  export PORT=8116 SPEC="$n"
  echo "=== ARM $tag $(date -u +%H:%M:%S) ==="
  bash "$LAUNCH_INT4" >/dev/null || { echo "[$tag] LAUNCH FAILED"; return 1; }
  for i in $(seq 1 200); do curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && break; sleep 5; done
  curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 || { echo "[$tag] ❌ NOT READY"; return 1; }
  grep -aoE "GPU KV cache size: [0-9,]+ tokens.*" "$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log | head -1)" | tail -1
  measure 8116 "$tag"
  stop_port 8116; sleep 25
}

run_int8_aiter () { # spec
  local n=$1; local tag="int8-aiter-mtp$n"
  stop_port 8115; wait_free || { echo "[$tag] GPU 不空"; return 1; }
  export PORT=8115 SPEC="$n"
  export PYTHONPATH="$REPO/aiter_patch"
  export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_ROCM_USE_AITER_RMSNORM=0
  export VLLM_ROCM_USE_AITER_MOE=0 VLLM_ROCM_USE_AITER_MLA=0 VLLM_ROCM_USE_AITER_MHA=0
  export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 VLLM_ROCM_USE_AITER_FP8BMM=0
  export VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=0 VLLM_ROCM_USE_AITER_TRITON_ROPE=0
  export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0 VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=0
  export VLLM_ROCM_USE_AITER_TRITON_GEMM=0
  unset VLLM_EXTRA_ARGS   # fastsafetensors OOMs on the int8 arm (51.75 GiB/die)
  echo "=== ARM $tag（AITER=$VLLM_ROCM_USE_AITER + 旁挂补丁）$(date -u +%H:%M:%S) ==="
  bash "$LAUNCH_INT8" >/dev/null || { echo "[$tag] LAUNCH FAILED"; return 1; }
  for i in $(seq 1 200); do curl -sf http://127.0.0.1:8115/health >/dev/null 2>&1 && break; sleep 5; done
  local srv; srv=$(ls -t /home/qiba/ai/logs/ornith397b/server-8115-*.log | head -1)
  curl -sf http://127.0.0.1:8115/health >/dev/null 2>&1 || { echo "[$tag] ❌ NOT READY"; grep -aoE "(ValueError|RuntimeError|AttributeError): .{0,150}" "$srv" | sort -u | head -3; return 1; }
  grep -aoE "Selected [A-Za-z0-9]*Int8ScaledMMLinearKernel|GPU KV cache size: [0-9,]+ tokens.*" "$srv" | sort | uniq -c
  measure 8115 "$tag"
  echo "--- 质量（aiter 臂数值不同，必须复核）---"
  docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG \
    -c "python3 -u /work/nll_probe.py 8115 ornith $tag" 2>&1 | tail -2
  stop_port 8115; sleep 25
}

run_int4 4
run_int4 5
run_int8_aiter 3
echo "=== next_batch done $(date -u +%FT%TZ) ==="
