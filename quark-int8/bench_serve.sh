#!/usr/bin/env bash
# Time the startup phases of the converted DSV4.1-INT4 server, then measure a
# single-stream baseline.
#
# The server itself is launched by the project-standard launcher, which is the
# single place holding the required patch mounts (the use_fnmatch CT patch and
# the MI250X GEMV MoE patch).  This script only wraps it with phase timing, so a
# patch can never be forgotten here and remembered there.
#
# Usage: PORT=8119 OUT=200 ./bench_serve.sh
set -uo pipefail

REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCHER=/home/qiba/ai/models/deepseek-ai/launcher/deepseek_v4.1_flash_ctint4w4a16_vllm_rocmnightly_256k_8119_dsv41_mi250dx8.sh
AI_HOME="${AI_HOME:-/home/qiba/ai}"
PORT="${PORT:-8119}"
OUT_TOKENS="${OUT:-200}"
PROMPT="${PROMPT:-Explain in detail how a transformer KV cache works.}"
MODEL_KEY="dsv41ctint4"
READY_TIMEOUT="${READY_TIMEOUT:-7200}"

[ -f "$LAUNCHER" ] || { echo "❌ launcher 不在：$LAUNCHER"; exit 1; }

T0=$(date +%s)
PORT="$PORT" bash "$LAUNCHER" || { echo "❌ launcher 起服失败"; exit 1; }

LOG="$(readlink -f "${AI_HOME}/logs/${MODEL_KEY}/server-${PORT}.current")"
echo "[bench] log -> $LOG"

# --- phase 1: safetensors -> GPU
LOAD_DONE=""
for _ in $(seq 1 "$READY_TIMEOUT"); do
  if grep -qa "Loading safetensors checkpoint shards: 100%" "$LOG" 2>/dev/null; then
    LOAD_DONE=$(date +%s); break
  fi
  if grep -qaE "KeyError|ValueError|RuntimeError|OutOfMemory" "$LOG" 2>/dev/null; then
    echo "[bench] 日志出现错误，停止等待"; break
  fi
  sleep 1
done
if [ -n "$LOAD_DONE" ]; then
  echo "[bench] phase1 权重装载     : $((LOAD_DONE - T0))s"
else
  echo "[bench] phase1 权重装载     : 未到达"
fi

# --- phase 2: init -> ready  (= CT->kernel repack + cudagraph capture)
READY=""
while [ $(( $(date +%s) - T0 )) -lt "$READY_TIMEOUT" ]; do
  if curl -s -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    READY=$(date +%s); break
  fi
  sleep 2
done

if [ -n "$READY" ]; then
  echo "[bench] phase2 init->ready   : $((READY - ${LOAD_DONE:-$READY}))s"
  echo "[bench] TOTAL 启动           : $((READY - T0))s"
else
  echo "[bench] phase2 init->ready   : FAILED / 超时"
  grep -naE "KeyError|ValueError|RuntimeError|OutOfMemory" "$LOG" | tail -5
  exit 1
fi

echo "[bench] --- 证据 ---"
grep -oaE "Using TritonW4A16LinearKernel|Using '[A-Z]*' WNA16 MoE backend\.|\[MI250_MOE_GEMV\][^\"]*" "$LOG" \
  | sort | uniq -c | head -6

python3 "$REPO/bench_infer.py" "$PORT" "$OUT_TOKENS" "$PROMPT"
echo "[bench] log: $LOG"
