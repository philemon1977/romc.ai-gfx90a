#!/usr/bin/env bash
# G 臂 = **交付态复测**：完全用 8117 启动脚本自己的默认值（不传任何 env 覆盖），
# 量出"脚本一跑就是这样"的最终数字：count n=5 + explain n=5（与 int4 臂同 prompt 对）
# + 步时分解。同时验证 VLLM_DISABLE_COMPILE_CACHE 默认 1 真的生效（日志应 0 载入 0 保存）。
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int8w8a8attn_aiter_vllm_rocm72_mtp5_256k_8117_ornith_mi250dx8.sh
PIDF=/home/qiba/ai/logs/ornith397b-8117.pid
exec > >(tee -a "$REPO/logs/int8aiter_armG.log") 2>&1
echo "=== arm G (交付态) start $(date -u +%FT%TZ) ==="

if [ -f "$PIDF" ]; then p=$(cat "$PIDF"); kill -TERM -"$p" 2>/dev/null && echo "已 TERM $p"; fi
rm -f "$PIDF"
for _ in $(seq 1 120); do
  nc -z 127.0.0.1 8117 2>/dev/null && { sleep 10; continue; }
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "${free:-0}" -ge 62 ] && { echo "显存已回收 (min free $free GiB)"; break; }
  sleep 10
done

# 不给任何 env：全用脚本默认（SPEC=5 / AITER=1 / no-pfx / DISABLE_COMPILE_CACHE=1）
env -u SPEC -u VLLM_ROCM_USE_AITER -u VLLM_DISABLE_COMPILE_CACHE PORT=8117 bash "$LAUNCH" || { echo "❌ LAUNCH FAILED"; exit 1; }
t0=$(date +%s)
for i in $(seq 1 240); do curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 && break; sleep 5; done
curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 || { echo "❌ NOT READY"; exit 1; }
echo "✅ ready 用时 $(( $(date +%s)-t0 ))s"
S=$(ls -t /home/qiba/ai/logs/ornith397b/server-8117-*.log | head -1)
echo "可见性检查（应 0 载入 0 保存 = 开关生效）：载入=$(grep -acE 'Directly load AOT' "$S") 保存=$(grep -acE 'saved AOT compiled' "$S")"
grep -aoE "Selected AiterInt8ScaledMMLinearKernel|Using [A-Za-z_']+ Int8 MoE backend|GPU KV cache size: [0-9,]+ tokens|Loading weights took [0-9.]+ seconds" "$S" | sort | uniq -c | sed 's/^/  /'

echo "--- count n=5 ---"
python3 "$REPO/measure_median.py" 8117 int8aiter-SHIP-count 5 256 count 2>&1 | sed 's/^/  /'
echo "--- explain n=5 ---"
python3 "$REPO/measure_median.py" 8117 int8aiter-SHIP-explain 5 256 explain 2>&1 | sed 's/^/  /'
echo "--- 步时分解 count ---"
python3 "$REPO/step_probe.py" 8117 int8aiter-SHIP 5 count 256 2>&1 | sed 's/^/  /'
echo "--- 步时分解 explain ---"
python3 "$REPO/step_probe.py" 8117 int8aiter-SHIP 5 explain 256 2>&1 | sed 's/^/  /'

p=$(cat "$PIDF" 2>/dev/null || true); [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null && echo "已停服 $p"
echo "=== arm G done $(date -u +%FT%TZ) ==="
