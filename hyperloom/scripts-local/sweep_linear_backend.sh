#!/usr/bin/env bash
# 扫 --linear-backend：目标那 60% 的 INT8 GEMM。
#
# 依据：profiling 归因显示 conc 64 的 decode 步里 60.09% 花在
#   ck::kernel_gemm_xdl_cshuffle_v3_multi_d（283 µs/次，240 次/步）
# 而 KernelConfig.linear_backend 默认 "auto" 就落到它上面。gfx90a 上有意义的取值：
#   auto / aiter / triton / torch   （其余为 CUDA 系）
#
# 负载与探针 B 的 conc 64 点一致（ISL/OSL 1024、128 prompts、--ignore-eos），
# 所以可直接与未改动的 473.78 tok/s 对比。
#
# 用法（容器内）：bash sweep_linear_backend.sh <outdir>
set -uo pipefail
OUT="${1:?outdir required}"
MODEL=/mnt/stripe-3mix-3t2/models/davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8
PORT=8890
mkdir -p "$OUT"
CSV="$OUT/sweep.csv"
echo "linear_backend,status,output_tput,mean_tpot_ms,mean_ttft_ms,duration_s" > "$CSV"

for LB in auto aiter triton torch; do
  echo "=== linear_backend=$LB ==="
  rm -rf /tmp/vllm-cache-lb
  setsid /usr/local/bin/vllm-wu1w serve "$MODEL" \
    --port $PORT --max-model-len 6144 \
    --trust-remote-code --language-model-only --quantization compressed-tensors \
    --safetensors-load-strategy eager \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --linear-backend "$LB" \
    > "$OUT/server-$LB.log" 2>&1 &
  SPID=$!
  echo "  server pid=$SPID"

  ok=0
  for i in $(seq 1 54); do
    code=$(curl -s -m 5 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null)
    [ "$code" = "200" ] && { ok=1; echo "  ready after ~$((i*10))s"; break; }
    kill -0 "$SPID" 2>/dev/null || { echo "  server 进程已退出"; break; }
    sleep 10
  done

  if [ $ok -ne 1 ]; then
    echo "  SERVER FAILED"
    echo "$LB,server_failed,,,," >> "$CSV"
    kill -KILL -- "-$SPID" 2>/dev/null; kill -KILL "$SPID" 2>/dev/null
    sleep 8; continue
  fi

  /opt/envs/vllm/bin/vllm bench serve \
    --model "$MODEL" --backend openai --base-url "http://127.0.0.1:$PORT" \
    --dataset-name random --random-input-len 1024 --random-output-len 1024 \
    --num-prompts 128 --max-concurrency 64 --ignore-eos \
    --save-result --result-dir "$OUT" --result-filename "lb-$LB.json" \
    > "$OUT/bench-$LB.log" 2>&1
  BRC=$?
  if [ $BRC -ne 0 ]; then
    echo "  BENCH FAILED rc=$BRC"
    echo "$LB,bench_failed,,,," >> "$CSV"
  else
    /opt/envs/vllm/bin/python3 - "$OUT/lb-$LB.json" "$LB" "$CSV" <<'PY'
import json, sys
r, lb, csv = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(r))
row = [lb, "ok",
       round(d.get("output_throughput", 0), 2),
       round(d.get("mean_tpot_ms", 0), 2),
       round(d.get("mean_ttft_ms", 0), 2),
       round(d.get("duration", 0), 1)]
print("  ", row)
open(csv, "a").write(",".join(str(x) for x in row) + "\n")
PY
  fi

  kill -TERM -- "-$SPID" 2>/dev/null; kill -TERM "$SPID" 2>/dev/null
  sleep 10
  kill -KILL -- "-$SPID" 2>/dev/null; kill -KILL "$SPID" 2>/dev/null
  sleep 5
done

echo "=== CSV ==="; cat "$CSV"
