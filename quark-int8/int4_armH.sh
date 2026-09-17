#!/usr/bin/env bash
# H 臂 = **int4 出厂臂同口径回测**（同会话、同 prompt、同 measure_median 口径 + 步时分解）。
# 目的：把"INT8-Attn 追平/反超 int4"从跨时段对比升级为**同条件对照**，并回答一个关键问题——
# int8 权重是 int4 的 2 倍，它是靠"步时更短"还是靠"MTP 接受率更高"赢的？
# int4 的 AOT 产物已存在 ⇒ 本臂会走"载入产物"类（与它历史上 89.5–90.2 那些跑法同类）✓
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
PIDF=/home/qiba/ai/logs/ornith397b-8116.pid
exec > >(tee -a "$REPO/logs/int4_armH.log") 2>&1
echo "=== arm H (int4 同口径) start $(date -u +%FT%TZ) ==="

if [ -f "$PIDF" ]; then p=$(cat "$PIDF"); kill -TERM -"$p" 2>/dev/null && echo "已 TERM $p"; fi
rm -f "$PIDF"
for _ in $(seq 1 120); do
  nc -z 127.0.0.1 8116 2>/dev/null && { sleep 10; continue; }
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "${free:-0}" -ge 62 ] && { echo "显存已回收 (min free $free GiB)"; break; }
  sleep 10
done

PORT=8116 bash "$LAUNCH" || { echo "❌ LAUNCH FAILED"; exit 1; }
t0=$(date +%s)
for i in $(seq 1 240); do curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && break; sleep 5; done
curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 || { echo "❌ NOT READY"; exit 1; }
echo "✅ ready 用时 $(( $(date +%s)-t0 ))s"
S=$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log | head -1)
echo "编译路径：载入=$(grep -acE 'Directly load AOT' "$S") 保存=$(grep -acE 'saved AOT compiled' "$S")"
grep -aoE "GPU KV cache size: [0-9,]+ tokens|Using [A-Za-z_']+ MoE backend|Loading weights took [0-9.]+ seconds" "$S" | sort | uniq -c | sed 's/^/  /'

echo "--- count n=5 ---"
python3 "$REPO/measure_median.py" 8116 int4-H-count 5 256 count 2>&1 | sed 's/^/  /'
echo "--- explain n=5 ---"
python3 "$REPO/measure_median.py" 8116 int4-H-explain 5 256 explain 2>&1 | sed 's/^/  /'
echo "--- 步时分解 count ---"
python3 "$REPO/step_probe.py" 8116 int4-H 5 count 256 2>&1 | sed 's/^/  /'
echo "--- 步时分解 explain ---"
python3 "$REPO/step_probe.py" 8116 int4-H 5 explain 256 2>&1 | sed 's/^/  /'

p=$(cat "$PIDF" 2>/dev/null || true); [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null && echo "已停服 $p"
echo "=== arm H done $(date -u +%FT%TZ) ==="
