#!/usr/bin/env bash
# Restart the int4 arm with vLLM 0.28.0's torch profiler enabled (--profiler-config,
# NOT the VLLM_TORCH_PROFILER_DIR env var, which 0.28.0 does not read for routes).
# 1) re-measure single-stream TPS (reproducibility of 41.59)
# 2) /start_profile -> 64-token decode -> /stop_profile
# 3) aggregate the trace into a per-kernel breakdown (Amdahl ceiling for any new kernel)
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
PIDF=/home/qiba/ai/logs/ornith397b-8116.pid
LOG="$REPO/logs/prof2.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== prof2 start $(date -u +%FT%TZ) ==="

# --- stop the currently running arm ---
if [ -f "$PIDF" ]; then
  _p=$(cat "$PIDF" 2>/dev/null || true)
  [ -n "${_p:-}" ] && kill -TERM -"$_p" 2>/dev/null && echo "已 TERM 会话组 $_p"
fi
for i in $(seq 1 60); do
  f=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  nc -z 127.0.0.1 8116 2>/dev/null || { [ "${f:-0}" -ge 62 ] && break; }
  sleep 10
done
rm -f "$PIDF"
echo "GPU 最空 die: ${f:-?} GiB"

# --- relaunch with the torch profiler attached to the API server ---
rm -rf "${REPO:?}/prof"/* 2>/dev/null
export VLLM_EXTRA_ARGS="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"$REPO/prof\",\"torch_profiler_with_stack\":false}"
echo "VLLM_EXTRA_ARGS=$VLLM_EXTRA_ARGS"
bash "$LAUNCH" || { echo "❌ LAUNCH FAILED"; exit 1; }

for i in $(seq 1 200); do
  curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && { echo "✅ ready (~$((i*5))s)"; break; }
  sleep 5
done
SRV=$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log | head -1)
if ! curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1; then
  echo "❌ NOT READY"; grep -aoE "(ValueError|RuntimeError|AssertionError|AttributeError): .{0,180}" "$SRV" | sort -u | head -5; exit 1
fi
grep -aoE "Loading weights took [0-9.]+ seconds|GPU KV cache size: [0-9,]+ tokens.*|Available KV cache memory: [0-9.]+ GiB" "$SRV" | tail -3

echo "--- 单流 TPS 复测（256 token, greedy, MTP）---"
python3 - <<'PY'
import json,time,urllib.request
body=json.dumps({'model':'ornith','prompt':'Explain in detail why int4 quantization reduces memory bandwidth pressure during decoding.','max_tokens':256,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8116/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=1800)); dt=time.time()-t0
u=d['usage']; print('[8116 int4+MTP 256K] 复测 SINGLE-STREAM %.2f tok/s (%d tok in %.1fs)'%(u['completion_tokens']/dt,u['completion_tokens'],dt))
PY

echo "--- start_profile ---"; curl -s -X POST http://127.0.0.1:8116/start_profile; echo
sleep 4
python3 - <<'PY'
import json,time,urllib.request
body=json.dumps({'model':'ornith','prompt':'Explain step by step why int4 weights reduce decode-time bandwidth.','max_tokens':96,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8116/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=900)); dt=time.time()-t0
u=d['usage']; print('[profile req] %d tok in %.1fs -> %.2f tok/s'%(u['completion_tokens'],dt,u['completion_tokens']/dt))
PY
sleep 3
echo "--- stop_profile ---"; curl -s -X POST http://127.0.0.1:8116/stop_profile; echo
for i in $(seq 1 60); do
  n=$(find "$REPO/prof" -type f 2>/dev/null | wc -l); [ "$n" -gt 0 ] && break; sleep 5
done
echo "--- trace 文件 ---"; find "$REPO/prof" -type f -printf '%s\t%p\n' 2>/dev/null | sort -rn | head -5
echo "--- 分析 ---"
python3 "$REPO/analyze_trace.py" "$REPO/prof" 22 2>&1 | tail -50
echo "=== prof2 done $(date -u +%FT%TZ) ==="
