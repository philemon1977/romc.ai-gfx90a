#!/usr/bin/env bash
# Attack the 27 ms/step fixed overhead.
# Step 1: SPEC=5 control median (same-session baseline for later single-variable A/Bs)
#         + ONE attempt to attach rocprofv3 to a worker PID to get kernel-level evidence
#         (torch profiler is unusable: --profiler-config kills EngineCore on this env).
set -uo pipefail
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
REPO=/home/qiba/ROCm.AI/quark-int8
LOG="$REPO/logs/fixed_overhead.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== fixed_overhead step1 start $(date -u +%FT%TZ) ==="

pidf=/home/qiba/ai/logs/ornith397b-8116.pid
[ -f "$pidf" ] && { p=$(cat "$pidf"); kill -TERM -"$p" 2>/dev/null; rm -f "$pidf"; }
for _ in $(seq 1 180); do
  busy=$(docker ps --format '{{.Names}}' | grep -c '^ornith-' || true)
  nc -z 127.0.0.1 8116 2>/dev/null && { sleep 15; continue; }
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "$busy" = "0" ] && [ "${free:-0}" -ge 62 ] && break
  sleep 15
done

export PORT=8116 SPEC=5
echo "=== 起服 SPEC=5（默认档 = 控制臂）$(date -u +%H:%M:%S) ==="
bash "$LAUNCH" >/dev/null || { echo "LAUNCH FAILED"; exit 1; }
for i in $(seq 1 200); do curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && { echo "✅ ready (~$((i*5))s)"; break; }; sleep 5; done
curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 || { echo "❌ NOT READY"; exit 1; }

python3 - <<'PY'
import json, statistics, time, urllib.request, secrets
tps, steps, toks = [], 0, 0
for rep in range(5):
    salt = secrets.token_hex(4)
    prompt = f"[salt {salt}] Explain in detail why int4 quantization reduces memory bandwidth pressure during decoding."
    body = json.dumps({'model': 'ornith', 'prompt': prompt, 'max_tokens': 256,
                       'temperature': 0, 'ignore_eos': True}).encode()
    req = urllib.request.Request('http://127.0.0.1:8116/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time(); d = json.load(urllib.request.urlopen(req, timeout=1800)); dt = time.time() - t0
    u = d['usage']; tps.append(u['completion_tokens'] / dt)
    print(f"  rep{rep+1}: {tps[-1]:.2f} tok/s ({u['completion_tokens']} tok in {dt:.1f}s)")
med = statistics.median(tps)
print(f"[CONTROL SPEC=5] MEDIAN {med:.2f} tok/s  min {min(tps):.2f} max {max(tps):.2f} spread {100*(max(tps)-min(tps))/med:.1f}%")
PY
curl -s localhost:8116/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{" | sed 's/^/  /'

echo "--- rocprofv3 attach 尝试 ---"
WPID=$(pgrep -f "VLLM::Worker_TP0" | head -1)
echo "Worker_TP0 pid=${WPID:-未找到}"
if [ -n "${WPID:-}" ]; then
  BEFORE=$(find /home/qiba "$REPO" -maxdepth 2 -newermt "-1 minute" -type f 2>/dev/null | wc -l)
  ( cd "$REPO/prof_attach" 2>/dev/null || { mkdir -p "$REPO/prof_attach"; cd "$REPO/prof_attach"; }; \
    ROCPROF_ATTACH_PID="$WPID" ROCPROF_ATTACH_DURATION=6000 \
    timeout 60 /opt/rocm-7.2.4/bin/rocprofv3-attach > "$REPO/prof_attach/attach.out" 2>&1 ) &
  ATT=$!
  sleep 2
  python3 - <<'PY'
import json,time,urllib.request
body=json.dumps({'model':'ornith','prompt':'Attach probe: count slowly.','max_tokens':32,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8116/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=600)); dt=time.time()-t0
print(f"  [attach 期间] {d['usage']['completion_tokens']} tok in {dt:.1f}s -> {d['usage']['completion_tokens']/dt:.2f} tok/s")
PY
  wait $ATT 2>/dev/null || true
  echo "--- attach 输出 ---"; cat "$REPO/prof_attach/attach.out" 2>/dev/null | head -6
  echo "--- 新产生的文件（可能是 trace）---"
  find "$REPO/prof_attach" /home/qiba -maxdepth 1 -newermt "-3 minutes" -type f 2>/dev/null | head -10
fi
echo "=== fixed_overhead step1 done $(date -u +%FT%TZ) ==="
