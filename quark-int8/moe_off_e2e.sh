#!/usr/bin/env bash
# End-to-end A/B for the integrated GEMV MoE (MI250_MOE_GEMV=1), on the current best recipe
# (SPEC=5 + --no-enable-prefix-caching), with the 0.1%-precision methodology.
# Baseline on the count prompt (5 reps, first discarded): 89.81 / 89.21 / 89.35 / 89.52 / 89.71
# Per-layer saving measured offline: 61.8 us => 3.71 ms/step => expected ~+11%.
set -uo pipefail
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
REPO=/home/qiba/ROCm.AI/quark-int8
IMG=vllm/vllm-openai-rocm:nightly
LOG="$REPO/logs/moe_gemv_off.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== moe_gemv_e2e start $(date -u +%FT%TZ) ==="

pidf=/home/qiba/ai/logs/ornith397b-8116.pid
[ -f "$pidf" ] && { p=$(cat "$pidf" 2>/dev/null || true); [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null; rm -f "$pidf"; }
for _ in $(seq 1 180); do
  nc -z 127.0.0.1 8116 2>/dev/null && { sleep 15; continue; }
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "${free:-0}" -ge 62 ] && break
  sleep 15
done

export PORT=8116 SPEC=5
export PYTHONPATH="$REPO/moe_gemv_patch"
export MI250_MOE_GEMV=0
export MI250_MOE_GEMV_DEBUG=1
echo "=== 起服（PYTHONPATH=$PYTHONPATH, MI250_MOE_GEMV=0）$(date -u +%H:%M:%S) ==="
bash "$LAUNCH" >/dev/null || { echo "LAUNCH FAILED"; exit 1; }
for i in $(seq 1 200); do curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && { echo "✅ ready (~$((i*5))s)"; break; }; sleep 5; done
SRV=$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log | head -1)
if ! curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1; then
  echo "❌ NOT READY"; grep -aoE "(ValueError|RuntimeError|AssertionError|ImportError): .{0,160}" "$SRV" | sort -u | head -4; exit 1
fi
echo "--- 补丁是否装上 ---"
grep -aoE "\[MI250_MOE_GEMV[^\"]{0,80}" "$SRV" | sort -u | head -3

echo "--- 单流 TPS（count prompt，5 次，丢弃首次；基线 89.6±0.3）---"
python3 "$REPO/measure_median.py" 8116 gemv-OFF 5 256 count
curl -s localhost:8116/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{" | sed 's/^/  /'
echo "--- takeover 计数（服务日志）---"
grep -aoE "\[MI250_MOE_GEMV takeover=[0-9]+ skip=[0-9]+[^\"]{0,40}" "$SRV" | tail -2

echo "--- 质量复核（NLL）---"
docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG \
  -c "python3 -u /work/nll_probe.py 8116 ornith moe-gemv-OFF" 2>&1 | tail -2

p2=$(cat "$pidf" 2>/dev/null || true); [ -n "${p2:-}" ] && kill -TERM -"$p2" 2>/dev/null && echo "已停服"
echo "=== moe_gemv_e2e done $(date -u +%FT%TZ) ==="
