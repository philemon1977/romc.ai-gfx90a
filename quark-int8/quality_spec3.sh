#!/usr/bin/env bash
# Formal quality measurement for the int4 factory recipe.
# The shipped default is SPEC=5, but vLLM 0.28's logprob reporting is broken at MTP depth 5
# (NLL ~12.9 == ln(vocab); see RESULT.md), so quality is measured at SPEC=3 (validated clean:
# NLL ~1.965) on the SAME launcher with only the depth changed -> a legitimate proxy.
# Records: NLL via both paths (repeat x2), speed on the count prompt, acceptance rate.
set -uo pipefail
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
REPO=/home/qiba/ROCm.AI/quark-int8
LOG="$REPO/logs/quality_spec3.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== quality_spec3 start $(date -u +%FT%TZ) ==="

pidf=/home/qiba/ai/logs/ornith397b-8116.pid
[ -f "$pidf" ] && { p=$(cat "$pidf" 2>/dev/null || true); [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null; rm -f "$pidf"; }
for _ in $(seq 1 120); do nc -z 127.0.0.1 8116 2>/dev/null && { sleep 15; continue; }
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "${free:-0}" -ge 62 ] && break; sleep 10; done

export PORT=8116 SPEC=3
echo "=== 起服（出厂 launcher，SPEC=3 仅改深度）$(date -u +%H:%M:%S) ==="
bash "$LAUNCH" >/dev/null || { echo "LAUNCH FAILED"; exit 1; }
for i in $(seq 1 80); do curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && break; sleep 10; done
curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 || { echo "未就绪"; exit 1; }
SRV=$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log | head -1)
grep -aoE "GPU KV cache size: [0-9,]+ tokens.*" "$SRV" | tail -1

echo "--- 质量：两条路径 ×2 次（探针自带闸门，NLL>8 会拒绝采信）---"
for r in 1 2; do
  echo "  [repeat $r]"
  python3 -u "$REPO/diag_nll.py" 8116 ornith "SPEC3-r$r" 2>&1 | grep -aE "NLL=|前 5"
  echo "  退出码=$?"
done

echo "--- 速度：count prompt，5 次丢弃首次 ---"
python3 "$REPO/measure_median.py" 8116 SPEC3-count 5 256 count
curl -s localhost:8116/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{" | sed 's/^/  /'

p2=$(cat "$pidf" 2>/dev/null || true); [ -n "${p2:-}" ] && kill -TERM -"$p2" 2>/dev/null
echo "=== quality_spec3 done $(date -u +%FT%TZ) ==="
