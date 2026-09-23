#!/bin/bash
# 等就绪 → 跑 decode-vs-prefill 自洽性检验 → 跑 prefill-only PPL。都只读 API，不改服务。
# 用法: bash run_decode_diag.sh [PORT]
set -u
PORT=${PORT:-8119}
R=/home/qiba/ROCm.AI/quark-int8
TS=$(date +%m%d_%H%M)
OUT=$R/logs/decode_diag_$TS.log
echo "=== $(date +%T) 等就绪（最多 20 min）→ $OUT"
for i in $(seq 1 120); do
  if curl -s -m 3 "http://127.0.0.1:${PORT}/v1/models" | grep -q '"id"'; then
    echo "✅ $(date +%T) 就绪"; break
  fi
  sleep 10
done
if ! curl -s -m 3 "http://127.0.0.1:${PORT}/v1/models" | grep -q '"id"'; then
  echo "❌ 未就绪，退出"; exit 1
fi

echo; echo "=== $(date +%T) [1/2] decode-vs-prefill 自洽性（短 prompt）==="
timeout 1200 python3 "$R/decode_prefill_consistency.py" --port "$PORT" \
  --out "$R/logs/decode_consistency_$TS.json" 2>&1 | tail -80

echo; echo "=== $(date +%T) [2/2] prefill-only PPL（自然文本，只做 prefill）==="
timeout 1800 python3 "$R/prefill_ppl.py" --port "$PORT" --max-tokens 600 \
  --out "$R/logs/prefill_ppl_$TS.json" 2>&1 | tail -70

echo; echo "=== $(date +%T) 诊断完毕（服务保持运行，未停）==="
