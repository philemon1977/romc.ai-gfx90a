#!/usr/bin/env bash
# E 臂 = 第三次冷启复跑 8117 出厂配方（AITER=1 SPEC=5 no-pfx），目的：
#   ① 跨重启分布：已有 109.36(A) / 103.22(D) 两个样本，再加一个
#   ② 步时/接受率分解（D 臂因正则漏 vllm: 前缀除零失败，已修）：把"更快"拆成
#      "每步更快"还是"每步接受更多"——裸 tok/s 分不开这两件事
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int8w8a8attn_aiter_vllm_rocm72_mtp5_256k_8117_ornith_mi250dx8.sh
PIDF=/home/qiba/ai/logs/ornith397b-8117.pid
exec > >(tee -a "$REPO/logs/int8aiter_armF.log") 2>&1
echo "=== arm D (repro) start $(date -u +%FT%TZ) ==="

if [ -f "$PIDF" ]; then p=$(cat "$PIDF"); kill -TERM -"$p" 2>/dev/null && echo "已 TERM $p"; fi
rm -f "$PIDF"
for _ in $(seq 1 120); do
  nc -z 127.0.0.1 8117 2>/dev/null && { sleep 10; continue; }
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "${free:-0}" -ge 62 ] && { echo "显存已回收 (min free $free GiB)"; break; }
  sleep 10
done

PORT=8117 SPEC=5 VLLM_ROCM_USE_AITER=1 VLLM_DISABLE_COMPILE_CACHE=1 bash "$LAUNCH" || { echo "❌ LAUNCH FAILED"; exit 1; }
t0=$(date +%s)
for i in $(seq 1 240); do curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 && break; sleep 5; done
curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 || { echo "❌ NOT READY"; exit 1; }
echo "✅ ready 用时 $(( $(date +%s)-t0 ))s"
S=$(ls -t /home/qiba/ai/logs/ornith397b/server-8117-*.log | head -1)
grep -aoE "Selected AiterInt8ScaledMMLinearKernel|GPU KV cache size: [0-9,]+ tokens|Loading weights took [0-9.]+ seconds" "$S" | sort | uniq -c | sed 's/^/  /'

echo "--- 复现：count n=5 ---"
python3 "$REPO/measure_median.py" 8117 int8aiter-mtp5-F 5 256 count 2>&1 | sed 's/^/  /'
echo "--- 步时分解：count ---"
python3 "$REPO/step_probe.py" 8117 int8aiter-mtp5 5 count 256 2>&1 | sed 's/^/  /'
echo "--- 步时分解：explain ---"
python3 "$REPO/step_probe.py" 8117 int8aiter-mtp5 5 explain 256 2>&1 | sed 's/^/  /'

p=$(cat "$PIDF" 2>/dev/null || true); [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null && echo "已停服 $p"
echo "=== arm D done $(date -u +%FT%TZ) ==="
