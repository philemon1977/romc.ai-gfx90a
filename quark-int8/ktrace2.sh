#!/usr/bin/env bash
# Correct way to get kernel-level evidence with rocprofv3's attach mode:
# the injected tool library (librocprofiler-sdk-tool.so) reads its config from the
# TARGET process environment, so the ROCPROF_* vars must be present when vLLM starts.
# Nothing is traced until attach injects the library, so startup (graph capture) stays
# out of the trace and the server's own performance is unaffected.
set -uo pipefail
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
REPO=/home/qiba/ROCm.AI/quark-int8
OUT="$REPO/prof_attach"
LOG="$REPO/logs/ktrace2.log"; mkdir -p "$REPO/logs" "$OUT"
exec > >(tee -a "$LOG") 2>&1
echo "=== ktrace2 start $(date -u +%FT%TZ) ==="

pidf=/home/qiba/ai/logs/ornith397b-8116.pid
[ -f "$pidf" ] && { p=$(cat "$pidf" 2>/dev/null || true); [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null; rm -f "$pidf"; }
for _ in $(seq 1 180); do
  busy=$(docker ps --format '{{.Names}}' | grep -c '^ornith-' || true)
  nc -z 127.0.0.1 8116 2>/dev/null && { sleep 15; continue; }
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "$busy" = "0" ] && [ "${free:-0}" -ge 62 ] && break
  sleep 15
done

rm -rf "$OUT"/ktr* "$OUT"/ktrace* 2>/dev/null
export PORT=8116 SPEC=5
# ↓ 这些必须在 vLLM 启动前导出：工具库 attach 时从目标进程环境读取
export ROCPROF_KERNEL_TRACE=1
export ROCPROF_RCCL_API_TRACE=1
export ROCPROF_OUTPUT_PATH="$OUT"
export ROCPROF_OUTPUT_FILE_NAME="ktr"
export ROCPROF_OUTPUT_FORMAT="csv"
export ROCPROF_DEMANGLE_KERNELS=1

echo "=== 起服 SPEC=5（带 ROCPROF 配置）$(date -u +%H:%M:%S) ==="
bash "$LAUNCH" >/dev/null || { echo "LAUNCH FAILED"; exit 1; }
for i in $(seq 1 200); do curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && { echo "✅ ready (~$((i*5))s)"; break; }; sleep 5; done
curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 || { echo "❌ NOT READY"; exit 1; }

echo "--- 预热（同一 prompt，便于与中位数对照）---"
python3 "$REPO/measure_median.py" 8116 warmup 1 128

WPID=$(pgrep -f "VLLM::Worker_TP0" | head -1)
echo "Worker_TP0 pid=${WPID:-未找到}"; [ -n "${WPID:-}" ] || exit 1

echo "--- attach 12 s + 同一 prompt 的 96 token 请求 ---"
( cd "$OUT" && timeout 180 /opt/rocm-7.2.4/bin/rocprofv3 --attach "$WPID" \
    --kernel-trace --attach-duration-msec 12000 \
    > "$OUT/rocprofv3_2.out" 2>&1 ) &
ATT=$!
sleep 3
python3 - <<'PY'
import json,time,urllib.request
body=json.dumps({'model':'ornith','prompt':'[salt deadbeef] Explain in detail why int4 quantization reduces memory bandwidth pressure during decoding, covering weights and KV cache.','max_tokens':96,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8116/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=900)); dt=time.time()-t0
u=d['usage']; print(f"  [trace 窗口] {u['completion_tokens']} tok in {dt:.2f}s -> {u['completion_tokens']/dt:.2f} tok/s")
PY
wait $ATT 2>/dev/null || true
echo "--- 产物 ---"; find "$OUT" -newermt "-3 minutes" -type f -printf "%s\t%p\n" 2>/dev/null | sort -rn | head -10
echo "--- 分析 ---"
python3 "$REPO/analyze_ktrace.py" "$OUT" 22 2>&1 | tail -45
echo "=== ktrace2 done $(date -u +%FT%TZ) ==="
