#!/usr/bin/env bash
# splitKV diagnostic, done right: the patch only writes its stats file every
# VLLM_ROCM_SPLITKV_PA_STATS_EVERY decisions (default 0 = never), so the batch-4 arm
# would have produced a false negative.
#
# The stats file carries reject_by_reason, and per the patch's own docstring a reject with
# max_query_len>1 IS the direct readout of "non-uniform decode step x layer count", i.e.
# how often the split-KV path could ever engage under MTP.
set -uo pipefail
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
REPO=/home/qiba/ROCm.AI/quark-int8
STATS="$REPO/logs/splitkv_stats.json"
LOG="$REPO/logs/splitkv_diag.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== splitkv_diag start $(date -u +%FT%TZ) ==="

pidf=/home/qiba/ai/logs/ornith397b-8116.pid
[ -f "$pidf" ] && { p=$(cat "$pidf" 2>/dev/null || true); [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null; rm -f "$pidf"; }
for _ in $(seq 1 180); do
  busy=$(docker ps --format '{{.Names}}' | grep -c '^ornith-' || true)
  nc -z 127.0.0.1 8116 2>/dev/null && { sleep 15; continue; }
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "$busy" = "0" ] && [ "${free:-0}" -ge 62 ] && break
  sleep 15
done
rm -f "$STATS"
export PORT=8116 SPEC=5
export VLLM_ROCM_SPLITKV_PA_STATS="$STATS"
export VLLM_ROCM_SPLITKV_PA_STATS_EVERY=400
echo "=== 起服（采纳配置 + splitKV 统计）$(date -u +%H:%M:%S) ==="
bash "$LAUNCH" >/dev/null || { echo "LAUNCH FAILED"; exit 1; }
for i in $(seq 1 200); do curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && { echo "✅ ready (~$((i*5))s)"; break; }; sleep 5; done
curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 || { echo "❌ NOT READY"; exit 1; }

python3 "$REPO/measure_median.py" 8116 splitkv-diag 5 256 count
echo "--- 统计文件（$STATS）---"
if [ -f "$STATS" ]; then cat "$STATS"; else echo "  文件不存在"; fi
echo "--- 服务日志里 splitKV 相关行 ---"
srv=$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log | head -1)
grep -aoiE "splitkv[^\"]{0,60}|rocm_splitkv[^\"]{0,40}|Cannot use ROCm custom paged attention[^\"]{0,40}" "$srv" | sort -u | head -5
echo "=== splitkv_diag done $(date -u +%FT%TZ) ==="
