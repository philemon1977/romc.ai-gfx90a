#!/usr/bin/env bash
# 探针 B：dense INT8 模型的并发扫描 —— 聚合吞吐在哪里饱和？
#
# 与 baseline 完全同配置的 server（8890），只变 --max-concurrency。
# 负载形状与 baseline 对齐：ISL 1024 / OSL 1024 / dataset random。
# prompt 数以"约 2–4 个 wave"为准，为了在预算内跑完（低并发下无法用 320 个 prompt）。
#
# 用法（容器内）：bash probe_conc_sweep.sh <outdir>
set -uo pipefail
OUT="${1:-/home/qiba/ROCm.AI/hyperloom/reports/models/qwen38-27b-w8a8-dense/probe}"
MODEL=/mnt/stripe-3mix-3t2/models/davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8
BASE=http://127.0.0.1:8890
mkdir -p "$OUT"
CSV="$OUT/conc_sweep.csv"
echo "conc,num_prompts,output_tput_tok_s,total_tput_tok_s,mean_tpot_ms,mean_ttft_ms,duration_s" > "$CSV"

# conc:num_prompts —— 低并发少给 prompt，否则单流要跑几十分钟
for pair in 1:2 8:16 32:64 64:128 128:128; do
  C="${pair%%:*}"; N="${pair##*:}"
  echo "=== conc=$C prompts=$N ==="
  R="$OUT/conc${C}.json"
  /opt/envs/vllm/bin/vllm bench serve \
    --model "$MODEL" \
    --backend openai \
    --base-url "$BASE" \
    --dataset-name random \
    --random-input-len 1024 --random-output-len 1024 \
    --num-prompts "$N" --max-concurrency "$C" \
    --ignore-eos \
    --save-result --result-dir "$OUT" --result-filename "conc${C}.json" \
    > "$OUT/conc${C}.log" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then echo "  FAILED rc=$RC (see conc${C}.log)"; continue; fi
  python3 - "$R" "$C" "$N" "$CSV" <<'PY'
import json, sys
r, c, n, csv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
d = json.load(open(r))
row = [c, n,
       round(d.get("output_throughput", 0), 2),
       round(d.get("total_token_throughput", 0), 2),
       round(d.get("mean_tpot_ms", 0), 2),
       round(d.get("mean_ttft_ms", 0), 2),
       round(d.get("duration", 0), 1)]
print("  ", row)
open(csv, "a").write(",".join(str(x) for x in row) + "\n")
PY
done
echo "=== CSV ==="; cat "$CSV"
