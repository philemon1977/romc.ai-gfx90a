#!/usr/bin/env bash
# 出厂态复测（前缀缓存 ON/align）：在 8117 上直接量，不重启服务。
# 目的：把"新默认"的数字固化下来（单流 + 多轮 + 数值硬门 + 冷/暖 TTFT）。
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
exec > >(tee -a "$REPO/logs/ship_on_verify.log") 2>&1
echo "=== 出厂态复测（前缀缓存 ON）$(date -u +%FT%TZ) ==="

for i in $(seq 1 240); do curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 && break; sleep 5; done
if ! curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1; then echo "❌ 8117 未就绪"; exit 1; fi
echo "✅ 8117 就绪"
L=$(ls -t /home/qiba/ai/logs/ornith397b/server-8117-*.log | head -1)
echo "  日志 $L"
grep -aoE "enable_prefix_caching=[A-Za-z]+|Mamba cache mode is set to '[a-z]+'|GPU KV cache size: [0-9,]+ tokens" "$L" | sort -u | sed 's/^/    /'

echo "--- V2 数值硬门（出厂态）---"
python3 "$REPO/reuse_consistency.py" 8117 ship-on 64 1e-2 2>&1 | sed 's/^/  /'
echo "--- 单流 count n=5 ---"
python3 "$REPO/measure_median.py" 8117 ship-on 5 256 count 2>&1 | tail -3 | sed 's/^/  /'
echo "--- 多轮（4 轮 × 1500 tok）---"
python3 "$REPO/multiturn_probe.py" 8117 ship-on 4 1500 2>&1 | sed 's/^/  /'
echo "--- 多轮（4 轮 × 64 tok）---"
python3 "$REPO/multiturn_probe.py" 8117 ship-on-short 4 64 2>&1 | sed 's/^/  /'
echo "--- 冷/暖 TTFT ---"
python3 "$REPO/prefix_probe.py" 8117 ship-on 2>&1 | grep -aE "冷请求|暖请求|hits|queries|命中率" | sed 's/^/  /'
echo "=== done $(date -u +%FT%TZ) ==="
