#!/usr/bin/env bash
# hyperloom-fa 容器内: AITER FA vs 基线后端 A/B (Qwen3.8-27B BF16, TP1, GPU7)
set -uo pipefail
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=7
M=/mnt/stripe-3mix-3t2/models/davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8
PY=/opt/envs/vllm/bin/python
PORT=8901

bench_arm () {
  local arm=$1; shift
  local extra_envs=("$@")
  echo "===== ARM: $arm (${extra_envs[*]:-none}) ====="
  env "${extra_envs[@]}" $PY -m vllm.entrypoints.openai.api_server \
    --model "$M" --tensor-parallel-size 1 --max-model-len 6144 \
    --gpu-memory-utilization 0.90 --port $PORT --language-model-only --quantization compressed-tensors \
    --safetensors-load-strategy eager \
    > /tmp/fa-arm-$arm.log 2>&1 &
  local SPID=$!
  for i in $(seq 1 150); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
    sleep 5
  done
  curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "SERVER FAIL $arm"; tail -5 /tmp/fa-arm-$arm.log; kill -9 $SPID 2>/dev/null; return 1; }
  grep -aoE "Using [A-Za-z_]*attention backend[^\"]*|backend[=: ]+[A-Z_]*ATTENTION[A-Z_]*" /tmp/fa-arm-$arm.log | tail -2
  for CONC in 8 32; do
    /opt/envs/vllm/bin/vllm bench serve --backend vllm --endpoint /v1/completions \
      --model "$M" --host 127.0.0.1 --port $PORT \
      --dataset-name random --random-input-len 4096 --random-output-len 128 \
      --num-prompts $((CONC*4)) --max-concurrency $CONC --request-rate inf \
      2>&1 | grep -E "Successful requests|Output token throughput|Median TPOT|Traceback|Error" | sed "s/^/conc$CONC | /"
  done
  kill -9 $SPID 2>/dev/null; pkill -9 -f "openai.api_server" 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null; pkill -9 -f "VLLM" 2>/dev/null; sleep 10
}

bench_arm baseline
bench_arm aiter_fa VLLM_ROCM_USE_AITER=1 VLLM_ATTENTION_BACKEND=ROCM_AITER_FA
echo AB_DONE
