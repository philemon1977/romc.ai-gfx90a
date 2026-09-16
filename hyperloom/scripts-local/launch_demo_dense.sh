#!/usr/bin/env bash
# 容器内运行: Hyperloom 3h demo — Qwen3.8-27B (dense, Qwen3_5 混合注意力), TP=2, no-kernel
# 全新一次性启动: 中途禁止 kill/resume (避免 enablement 幻影门)
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
RUNLOG="${LOGDIR}/optimize-dense-${STAMP}.log"

cd "$REPO_ROOT"
setsid nohup python3 -m hyperloom.inference_optimizer.cli optimize \
  --model "$MODEL_PATH" \
  --framework vllm \
  --model-class dense \
  --tp 2 \
  --conc 64 --isl 1024 --osl 1024 \
  --precision bf16 \
  --target-gain 30 \
  --max-hours 3 \
  --max-minutes-framework-pct 0.50 \
  --max-minutes-sweep-pct 0.01 \
  --no-kernel \
  --no-enable-conc-sweep \
  --no-enable-roofline \
  --launch-info-file "${LOGDIR}/launch-info-dense-${STAMP}.json" \
  > "$RUNLOG" 2>&1 &
PID=$!
echo "OPTIMIZER_PID=$PID"
echo "RUNLOG=$RUNLOG"
sleep 25
if kill -0 $PID 2>/dev/null; then echo "ALIVE after 25s"; else echo "DIED early"; fi
grep -aE "session_dir=|session_id=" "$RUNLOG" | tail -2
tail -4 "$RUNLOG"
