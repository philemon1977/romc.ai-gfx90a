#!/usr/bin/env bash
# Final recipe verification on the ADOPTED config (SPEC=5 + --no-enable-prefix-caching):
# report single-stream TPS under all three workload regimes, each with the 0.1%-precision
# methodology (fixed prompt where applicable, first request excluded).
set -uo pipefail
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
REPO=/home/qiba/ROCm.AI/quark-int8
LOG="$REPO/logs/final_recipe.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== final_recipe start $(date -u +%FT%TZ) ==="

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
echo "=== 起服（采纳后配置）$(date -u +%H:%M:%S) ==="
bash "$LAUNCH" >/dev/null || { echo "LAUNCH FAILED"; exit 1; }
for i in $(seq 1 200); do curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && { echo "✅ ready (~$((i*5))s)"; break; }; sleep 5; done
curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 || { echo "❌ NOT READY"; exit 1; }
srv=$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log | head -1)
grep -aoE "GPU KV cache size: [0-9,]+ tokens.*|Mamba cache mode is set to '[a-z]+'|Add [0-9]+ padding layers[^\"]{0,30}" "$srv" | sort -u | sed 's/^/  /'

for m in count explain salt; do
  echo "--- mode=$m ---"
  python3 "$REPO/measure_median.py" 8116 "final-$m" 5 256 "$m"
done
curl -s localhost:8116/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{" | sed 's/^/  /'
echo "=== final_recipe done $(date -u +%FT%TZ) ==="
