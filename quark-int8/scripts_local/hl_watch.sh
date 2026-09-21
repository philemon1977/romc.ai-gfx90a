#!/usr/bin/env bash
# 读 Hyperloom 会话状态（不占卡；一切 python 都在容器里跑，因为 hyperloom 包在宿主机不可导入）
# 用法：bash quark-int8/scripts_local/hl_watch.sh [容器名]
set -uo pipefail
CTR="${1:-hyperloom-local}"
RUN_DIR=/home/qiba/ROCm.AI/hyperloom/session/optimizer_runs
HL=/home/qiba/ROCm.AI/hyperloom
[ -f "$RUN_DIR/last_launch.env" ] || { echo "没有 last_launch.env（会话还没起来？）"; exit 1; }
# shellcheck disable=SC1091
. "$RUN_DIR/last_launch.env"
: "${SESSION_DIR:?last_launch.env 里没有 SESSION_DIR（launch_health.sh 未跑成？）}"
echo "SESSION_DIR=$SESSION_DIR"
[ -d "$SESSION_DIR" ] || { echo "会话目录还不存在：$SESSION_DIR"; exit 1; }
echo "--- read_optimizer_state ---"
docker exec -e PYTHONPATH=$HL "$CTR" python3 \
  "$HL/hyperloom/inference_optimizer/tools/read_optimizer_state.py" "$SESSION_DIR" 2>&1 | tail -60
echo "--- event_counts ---"
docker exec -e PYTHONPATH=$HL "$CTR" python3 \
  "$HL/hyperloom/inference_optimizer/tools/event_counts.py" "$SESSION_DIR" 2>&1 | tail -25
echo "--- 进程 ---"
docker exec "$CTR" bash -lc 'ps -eo pid,etime,args | grep -E "inference_optimizer.cli optimize" | grep -v grep | head -2 | cut -c1-120 || echo "  optimizer 进程不在（会话已结束？）"'
docker exec "$CTR" bash -lc 'pgrep -fa "vllm serve" | head -2 | cut -c1-120 || echo "  无 vllm serve（候选间隙或已停）"'
