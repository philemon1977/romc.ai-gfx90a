#!/usr/bin/env bash
# hyperloom-srv 容器内：INT8 W8A8 Qwen3.8-27B 会话（gfx90a 先验注入 + 生产验证过的 serving env）
# 一次性干净启动 —— 中途禁止 kill/resume
set -uo pipefail
export REPO_ROOT="/home/qiba/ROCm.AI/hyperloom"
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
_dotenv_prev="$(export -p | grep -v -e '=""$' -e "=''\$")"
set -a; . "${REPO_ROOT}/.env"; set +a
eval "$_dotenv_prev"; unset _dotenv_prev
export USER_DATA_PATH="${USER_DATA_PATH:?}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HYPERLOOM_MODEL_ARCH_FILE="${REPO_ROOT}/session-priors/model_arch.int8-27b.json"
ulimit -Sn 65536 || true

# IR-2: install.sh 必须在启动 shell 里 exit 0（幂等重跑，多数已装）
bash "${REPO_ROOT}/hyperloom/inference_optimizer/assets/install.sh" >/tmp/install-$(date +%s).log 2>&1
RC=$?; echo "install.sh rc=$RC"
[ $RC -ne 0 ] && tail -5 /tmp/install-*.log | tail -3 && exit 1
. "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

LOGDIR="${USER_DATA_PATH}/launch"; mkdir -p "$LOGDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUNLOG="${LOGDIR}/optimize-int8-${STAMP}.log"

cd "$REPO_ROOT"
setsid nohup python3 -m hyperloom.inference_optimizer.cli optimize \
  --model "$MODEL_PATH" \
  --framework vllm \
  --model-class dense \
  --tp 1 --ep 1 \
  --conc 64 --isl 1024 --osl 1024 \
  --precision bf16 \
  --target-gain 30 \
  --max-hours 3 \
  --max-minutes-framework-pct 0.50 \
  --max-minutes-sweep-pct 0.01 \
  --no-kernel --no-enable-conc-sweep --no-enable-roofline \
  --server-args "--trust-remote-code --language-model-only --quantization compressed-tensors --safetensors-load-strategy eager --compilation-config {\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}" \
  --extra-env VLLM_ROCM_USE_AITER=1 \
  --extra-env VLLM_ROCM_USE_AITER_LINEAR=1 \
  --extra-env VLLM_BIN=/usr/local/bin/vllm-wu1w \
  --extra-env VLLM_CACHE_ROOT=/tmp/vllm-cache-hl \
  --launch-info-file "${LOGDIR}/launch-info-int8-${STAMP}.json" \
  > "$RUNLOG" 2>&1 &
PID=$!
echo "SESSION_PID=$PID"; echo "RUNLOG=$RUNLOG"
sleep 30
kill -0 $PID 2>/dev/null && echo "ALIVE after 30s" || echo "DIED"
grep -aE "session_dir=|HYPERLOOM_LAUNCH" "$RUNLOG" | tail -1
tail -3 "$RUNLOG"
