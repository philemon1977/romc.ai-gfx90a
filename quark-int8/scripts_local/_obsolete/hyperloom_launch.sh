#!/bin/bash
# Hyperloom 启动（在容器内执行）：IR-2 = install.sh + source runtime env，同一 shell 再 launch
set -uo pipefail
cd /home/qiba/ROCm.AI/hyperloom
export PYTHONPATH=/home/qiba/ROCm.AI/hyperloom
export HYPERLOOM_IMAGE=rocm-ai/vllm:glm53-int4-gfx90a-0918   # 用我们烘的镜像（含 7 补丁）
export USER_DATA_PATH=/home/qiba/ROCm.AI/hyperloom/session

echo "=== IR-2: install.sh $(date +%T) ==="
bash hyperloom/inference_optimizer/assets/install.sh 2>&1 | tail -5
echo "=== IR-2: source kernel-agent.env.sh ==="
source session/runtime/kernel-agent.env.sh && echo "  sourced ✓ (MAGPIE_PYTHON=$MAGPIE_PYTHON)"

RUN_TAG="glm53-int4-kernel-$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$USER_DATA_PATH/optimizer_runs"; mkdir -p "$RUN_DIR"
RUN_LOG="$RUN_DIR/run_${RUN_TAG}.log"; PID_FILE="$RUN_DIR/run_${RUN_TAG}.pid"
LAUNCH_INFO="$RUN_DIR/launch_${RUN_TAG}.json"

echo "=== launch optimize $(date +%T) ==="
echo "  model   : /mnt/kioxia-cm6-3t8/ai/models/ZhipuAI/GLM-5.3-CT-Int4-W4A16"
echo "  image   : $HYPERLOOM_IMAGE"
echo "  tp/ep   : 8/1   precision: w4a16"
echo "  workload: isl=1024 osl=1024 conc=8 (+conc sweep 1,4,8,16,32)"
echo "  target  : 最大化 TPS，下限 --target-tput 30   预算 3h"
echo "  log     : $RUN_LOG"

nohup python3 -m hyperloom.inference_optimizer.cli optimize \
  --model /mnt/kioxia-cm6-3t8/ai/models/ZhipuAI/GLM-5.3-CT-Int4-W4A16 \
  --framework vllm --tp 8 --ep 1 --precision w4a16 \
  --conc 8 --isl 1024 --osl 1024 \
  --max-hours 3 --target-tput 30 --tick-interval-sec 30 \
  --enable-conc-sweep --conc-sweep-concs 1,4,8,16,32 \
  --launch-info-file "$LAUNCH_INFO" \
  --server-args "--max-num-batched-tokens 2048 --max-num-seqs 32 --max-cudagraph-capture-size 8 --max-model-len 32768 --decode-context-parallel-size 8" \
  > "$RUN_LOG" 2>&1 &
echo $! > "$PID_FILE"
sleep 20
echo "  PID=$(cat $PID_FILE)"
echo "  --- 前 25 行日志 ---"
tail -25 "$RUN_LOG"
