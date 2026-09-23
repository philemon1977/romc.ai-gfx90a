#!/bin/bash
# 统一评测 battery（服务已在 8119）：事实召回 + eval_quality_ab（固定题/针尖/GSM8K16/复读率/tok·s^-1）
set -u
LBL=$1
cd /home/qiba/ROCm.AI/quark-int8
OUT=logs/quality_${LBL}_$(date +%m%d_%H%M).json
echo "=== [battery:$LBL] 事实召回 $(date +%T) ==="
python3 fact_recall_probe.py 8119 /models
echo "=== [battery:$LBL] eval_quality_ab ==="
python3 eval_quality_ab.py --port 8119 --label "$LBL" --n-gsm8k 16 \
  --gsm8k /home/qiba/ROCm.AI/quark-int8/refs/gsm8k_sample.jsonl --out "$OUT"
python3 - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("SUMMARY", json.dumps({k: v for k, v in d.items() if isinstance(v, (int, float, str))}, ensure_ascii=False))
PY
echo "RESULT_JSON=$OUT"
