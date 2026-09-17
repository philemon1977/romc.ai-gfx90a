#!/usr/bin/env bash
# MTP depth on the int4 arm — the only order-of-magnitude lever for single-stream TPS
# (weight bytes were proven irrelevant: §3 of docs/MI250X-AITER-INT4-内核复核-2026-09-17.md).
#
# Gate check: `bench/ledger.py gate "SPEC n=2"` = PASS. The existing rejection L9
# ("加深 MTP n=4/5/8/12/16") is llama.cpp/GLM-5.3-Flash and says n=2 is the peak, so
# n=2 must be measured here (vLLM + Qwen3.5-MoE MTP) rather than assumed.
#
# Arms: SPEC=1 (control, brackets run-to-run noise in this session), 2, 3.
# Waits for the GPU to be free first (single-tenant box; the int8 A/B runs before it).
set -uo pipefail
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
REPO=/home/qiba/ROCm.AI/quark-int8
LOG="$REPO/logs/mtp_depth.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== mtp_depth start $(date -u +%FT%TZ) ==="

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

run_arm () {  # spec_depth
  local n=$1
  local pidf=/home/qiba/ai/logs/ornith397b-8116.pid
  [ -f "$pidf" ] && { p=$(cat "$pidf"); kill -TERM -"$p" 2>/dev/null; rm -f "$pidf"; }
  wait_free || { echo "[SPEC=$n] GPU 一直不空"; return 1; }
  export PORT=8116 SPEC="$n"
  echo "=== ARM SPEC=$n $(date -u +%H:%M:%S) ==="
  bash "$LAUNCH" || { echo "[SPEC=$n] LAUNCH FAILED"; return 1; }
  for i in $(seq 1 200); do
    curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && { echo "[SPEC=$n] ✅ ready (~$((i*5))s)"; break; }
    sleep 5
  done
  local srv; srv=$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log | head -1)
  if ! curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1; then
    echo "[SPEC=$n] ❌ NOT READY："
    grep -aoE "(ValueError|RuntimeError|AssertionError|AttributeError): .{0,170}" "$srv" | sort -u | head -4
    return 1
  fi
  grep -aoE "GPU KV cache size: [0-9,]+ tokens.*" "$srv" | tail -1
  python3 - <<PY
import json,time,urllib.request
body=json.dumps({'model':'ornith','prompt':'Explain in detail why int4 quantization reduces memory bandwidth pressure during decoding.','max_tokens':256,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8116/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=1800)); dt=time.time()-t0
u=d['usage']
print('[SPEC=$n] SINGLE-STREAM %.2f tok/s (%d tok in %.1fs)'%(u['completion_tokens']/dt,u['completion_tokens'],dt))
PY
  curl -s localhost:8116/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{|spec_decode_num_accepted_tokens_per_pos" | sed "s/^/  [SPEC=$n] /"
  local p2; p2=$(cat "$pidf" 2>/dev/null || true)
  [ -n "${p2:-}" ] && kill -TERM -"$p2" 2>/dev/null && echo "[SPEC=$n] 已停服"
  sleep 25
}

for n in 1 2 3; do run_arm "$n"; done
echo "=== mtp_depth done $(date -u +%FT%TZ) ==="
