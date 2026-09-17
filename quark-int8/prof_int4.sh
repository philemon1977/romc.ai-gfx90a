#!/usr/bin/env bash
# Launch the int4 launcher itself (project env vllm 0.28.0), measure single-stream TPS
# (validates the 8116 arm end-to-end), then torch-profile a 64-token decode to get the
# per-kernel time breakdown. That breakdown is the Amdahl ceiling for any custom kernel.
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
LOG="$REPO/logs/prof_int4.log"; mkdir -p "$REPO/logs" "$REPO/prof"
exec > >(tee -a "$LOG") 2>&1
echo "=== prof_int4 start $(date -u +%FT%TZ) ==="

export VLLM_TORCH_PROFILER_DIR="$REPO/prof"
rm -rf "${REPO:?}/prof"/* 2>/dev/null

bash "$LAUNCH" || { echo "❌ LAUNCH FAILED"; exit 1; }

for i in $(seq 1 200); do
  curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && { echo "✅ ready (~$((i*5))s)"; break; }
  sleep 5
done
if ! curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1; then
  echo "❌ NOT READY — 失败原因："
  SRV=$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log 2>/dev/null | head -1)
  grep -oE "(ValueError|RuntimeError|NotImplementedError|AssertionError|AttributeError|ImportError): .{0,200}" "$SRV" | sort -u | head -6
  echo "--- 失败关键词扫描 ---"
  grep -cE "RuntimeError|initialization failed|EMULATION" "$SRV"
  exit 1
fi

SRV=$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log 2>/dev/null | head -1)
echo "--- 关键启动行（$SRV）---"
grep -oE "Resolved architecture: [A-Za-z0-9_]+|Using [A-Za-z_']+ MoE backend[^.]*\.|Loading weights took [0-9.]+ seconds|GPU KV cache size: [0-9,]+ tokens.*|Available KV cache memory: [0-9.]+ GiB|Selected [A-Za-z0-9]+ for [A-Za-z0-9_.]+" "$SRV" | tail -8

echo "--- 单流 TPS（256 token, greedy, MTP）---"
python3 - <<'PY'
import json,time,urllib.request
body=json.dumps({'model':'ornith','prompt':'Explain in detail why int4 quantization reduces memory bandwidth pressure during decoding.','max_tokens':256,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8116/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=1800)); dt=time.time()-t0
u=d['usage']; print('[8116 int4+MTP 256K] SINGLE-STREAM %.2f tok/s (%d tok in %.1fs)'%(u['completion_tokens']/dt,u['completion_tokens'],dt))
PY
curl -s localhost:8116/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{" | sed 's/^/  /'

echo "--- start_profile ---"
curl -s -X POST http://127.0.0.1:8116/start_profile; echo
sleep 4
python3 - <<'PY'
import json,time,urllib.request
body=json.dumps({'model':'ornith','prompt':'Count from one to sixty in words, one per line.','max_tokens':64,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8116/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=900)); dt=time.time()-t0
u=d['usage']; print('[profile req] %d tok in %.1fs -> %.2f tok/s'%(u['completion_tokens'],dt,u['completion_tokens']/dt))
PY
sleep 3
echo "--- stop_profile ---"
curl -s -X POST http://127.0.0.1:8116/stop_profile; echo
for i in $(seq 1 40); do
  n=$(ls -1 "$REPO/prof" 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && break; sleep 5
done
echo "--- trace 文件 ---"; find "$REPO/prof" -name '*.json*' -printf '%s\t%p\n' 2>/dev/null | sort -rn | head -10
echo "=== prof_int4 done $(date -u +%FT%TZ) ==="
