#!/usr/bin/env bash
# 容器内运行: 启动 Hyperloom 3h FRAMEWORK_AGENT demo (Qwen3.8-27B, TP=2, no-kernel)
set -uo pipefail
export REPO_ROOT="/home/qiba/ROCm.AI/hyperloom"
_dotenv_prev="$(export -p | grep -v -e '=""$' -e "=''\$")"
set -a; . "${REPO_ROOT}/.env"; set +a
eval "$_dotenv_prev"; unset _dotenv_prev
export USER_DATA_PATH="${USER_DATA_PATH:?}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
. "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
ulimit -Sn 65536 || true

LOGDIR="${USER_DATA_PATH}/launch"
mkdir -p "$LOGDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUNLOG="${LOGDIR}/optimize-${STAMP}.log"

cd "$REPO_ROOT"
setsid nohup python3 -m hyperloom.inference_optimizer.cli optimize \
  --model "$MODEL_PATH" \
  --framework vllm \
  --model-class moe_swa \
  --tp 2 --ep 2 \
  --server-args "--distributed-executor-backend mp" \
  --conc 64 --isl 1024 --osl 1024 \
  --precision bf16 \
  --target-gain 30 \
  --max-hours 3 \
  --max-minutes-framework-pct 0.50 \
  --max-minutes-sweep-pct 0.01 \
  --no-kernel \
  --no-enable-conc-sweep \
  --no-enable-roofline \
  --launch-info-file "${LOGDIR}/launch-info-${STAMP}.json" \
  > "$RUNLOG" 2>&1 &
PID=$!
echo "OPTIMIZER_PID=$PID"
echo "RUNLOG=$RUNLOG"
echo "LAUNCH_INFO=${LOGDIR}/launch-info-${STAMP}.json"
sleep 20
if kill -0 $PID 2>/dev/null; then echo "ALIVE after 20s"; else echo "DIED early"; fi
tail -15 "$RUNLOG"
