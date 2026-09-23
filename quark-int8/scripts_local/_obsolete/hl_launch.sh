#!/usr/bin/env bash
# Hyperloom optimize 启动包装（2026-09-21）
#
# 为什么不用 skill 自带的 scripts/launch.sh：它把 OPT_FLAGS 直接无引号展开
# （${OPT_FLAGS:-}），实测 '--a "--b c d" --e' 会被拆成 ["--b] [c] [d"] ⇒
# --server-args 这种"值里含空格"的 flag 必然传坏。本包装用数组传参。
# 其余（env 链、setsid nohup、日志/PID/launch-info/last_launch.env）与 launch.sh 一致。
#
# 用法：SKILL_DIR=/home/qiba/.dsh/skills/hyperloom-workload-optimizer bash quark-int8/scripts_local/hl_launch.sh
set -euo pipefail

SKILL_DIR="${SKILL_DIR:?请设置 SKILL_DIR 指向 hyperloom-workload-optimizer skill 目录}"
# shellcheck source=/dev/null
. "${SKILL_DIR}/scripts/_env.sh"

SERVER_ARGS="${SERVER_ARGS:?SERVER_ARGS 缺失（写进 workload.env）}"
RUN_TAG="$(basename "$MODEL_PATH")-$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${RUN_DIR}/run_${RUN_TAG}.log"
PID_FILE="${RUN_DIR}/run_${RUN_TAG}.pid"
LAUNCH_INFO_FILE="${RUN_DIR}/launch_${RUN_TAG}.json"

# OPT_FLAGS 只放"值里不含空格"的 flag（本仓 workload.env 已如此约定）
read -r -a OPT_ARR <<< "${OPT_FLAGS:-}"

setsid nohup "$PYTHON" -m hyperloom.inference_optimizer.cli --verbose optimize \
  --model "$MODEL_PATH" \
  --framework "$FRAMEWORK" \
  --tp "$TP" \
  --ep "$EP" \
  --conc "$CONC" \
  --isl "$ISL" \
  --osl "$OSL" \
  --precision "$PRECISION" \
  --max-hours "$MAX_HOURS" \
  --target-gain "$TARGET_GAIN" \
  --tick-interval-sec 30 \
  --launch-info-file "$LAUNCH_INFO_FILE" \
  --server-args "$SERVER_ARGS" \
  "${OPT_ARR[@]}" \
  > "$RUN_LOG" 2>&1 < /dev/null &

echo $! > "$PID_FILE"

cat > "$LAST_LAUNCH_ENV" <<EOF
export RUN_TAG="${RUN_TAG}"
export RUN_LOG="${RUN_LOG}"
export PID_FILE="${PID_FILE}"
export LAUNCH_INFO_FILE="${LAUNCH_INFO_FILE}"
EOF

echo "run_tag=${RUN_TAG}"
echo "run_log=${RUN_LOG}"
echo "launch_info_file=${LAUNCH_INFO_FILE}"
echo "server_args=${SERVER_ARGS}"
echo "opt_flags=${OPT_FLAGS:-}"
echo "next=bash ${SKILL_DIR}/scripts/launch_health.sh"
