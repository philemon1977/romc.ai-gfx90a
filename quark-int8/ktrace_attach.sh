#!/usr/bin/env bash
# Kernel-level evidence for the ~27 ms/step fixed overhead: attach rocprofv3 to one
# TP worker while a decode runs, then aggregate the kernel trace.
# torch profiler is not usable here (--profiler-config kills EngineCore on this env),
# but rocprofv3's own attach mode works:  rocprofv3 --attach <pid> --kernel-trace ...
set -uo pipefail
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
REPO=/home/qiba/ROCm.AI/quark-int8
OUT="$REPO/prof_attach"
LOG="$REPO/logs/ktrace_attach.log"; mkdir -p "$REPO/logs" "$OUT"
exec > >(tee -a "$LOG") 2>&1
echo "=== ktrace_attach start $(date -u +%FT%TZ) ==="

pidf=/home/qiba/ai/logs/ornith397b-8116.pid
[ -f "$pidf" ] && { p=$(cat "$pidf" 2>/dev/null || true); [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null; rm -f "$pidf"; }
for _ in $(seq 1 180); do
  busy=$(docker ps --format '{{.Names}}' | grep -c '^ornith-' || true)
  nc -z 127.0.0.1 8116 2>/dev/null && { sleep 15; continue; }
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "$busy" = "0" ] && [ "${free:-0}" -ge 62 ] && break
  sleep 15
done

export PORT=8116 SPEC=5
echo "=== 起服 SPEC=5 $(date -u +%H:%M:%S) ==="
bash "$LAUNCH" >/dev/null || { echo "LAUNCH FAILED"; exit 1; }
for i in $(seq 1 200); do curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && { echo "✅ ready (~$((i*5))s)"; break; }; sleep 5; done
curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 || { echo "❌ NOT READY"; exit 1; }

echo "--- 预热一个 128 token 请求（让 graph 就位）---"
python3 "$REPO/measure_median.py" 8116 warmup 1 128

WPID=$(pgrep -f "VLLM::Worker_TP0" | head -1)
echo "Worker_TP0 pid=${WPID:-未找到}"
[ -n "${WPID:-}" ] || exit 1

rm -rf "$OUT"/ktrace* 2>/dev/null
echo "--- 附着 rocprofv3（12 s 窗口，同时打一个 96 token 请求）---"
( cd "$OUT" && timeout 180 /opt/rocm-7.2.4/bin/rocprofv3 --attach "$WPID" \
    --kernel-trace --attach-duration-msec 12000 -f csv -o "$OUT/ktrace" \
    > "$OUT/rocprofv3.out" 2>&1 ) &
ATT=$!
sleep 3
python3 - <<'PY'
import json,time,urllib.request
body=json.dumps({'model':'ornith','prompt':'Trace window: count slowly from one to ninety.','max_tokens':96,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8116/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=900)); dt=time.time()-t0
u=d['usage']; print(f"  [trace 窗口内] {u['completion_tokens']} tok in {dt:.2f}s -> {u['completion_tokens']/dt:.2f} tok/s")
PY
wait $ATT 2>/dev/null || true
echo "--- rocprofv3 尾部输出 ---"; tail -5 "$OUT/rocprofv3.out" 2>/dev/null
echo "--- 产物 ---"; ls -la "$OUT"/ | grep -i ktrace | head -8
echo "--- 分析 ---"
python3 "$REPO/analyze_ktrace.py" "$OUT" 22 2>&1 | tail -45
echo "=== ktrace_attach done $(date -u +%FT%TZ) ==="
