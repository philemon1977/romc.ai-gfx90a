#!/usr/bin/env bash
# 恢复 INT8 会话（容器重建后 HIP/ROCR 冲突已消除；同一 session dir，不开新会话）
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
bash "${REPO_ROOT}/hyperloom/inference_optimizer/assets/install.sh" >/tmp/install-resume.log 2>&1 || { echo INSTALL_FAIL; tail -5 /tmp/install-resume.log; exit 1; }
. "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
SESSION_DIR="/home/qiba/ROCm.AI/hyperloom/session/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8/20260914T134902Z-244d9455"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUNLOG="${USER_DATA_PATH}/launch/optimize-int8-resume-${STAMP}.log"
cd "$REPO_ROOT"
setsid nohup python3 -m hyperloom.inference_optimizer.cli optimize \
  --resume-from "$SESSION_DIR" \
  --model "$MODEL_PATH" --framework vllm --model-class dense \
  --tp 1 --ep 1 --conc 64 --isl 1024 --osl 1024 --precision bf16 \
  --target-gain 30 --max-hours 3 \
  --max-minutes-framework-pct 0.50 --max-minutes-sweep-pct 0.01 \
  --no-kernel --no-enable-conc-sweep --no-enable-roofline \
  --server-args "--trust-remote-code --language-model-only --quantization compressed-tensors --safetensors-load-strategy eager --compilation-config {\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}" \
  --extra-env VLLM_ROCM_USE_AITER=1 --extra-env VLLM_ROCM_USE_AITER_LINEAR=1 \
  --extra-env VLLM_BIN=/usr/local/bin/vllm-wu1w \
  --extra-env VLLM_CACHE_ROOT=/tmp/vllm-cache-hl \
  > "$RUNLOG" 2>&1 &
echo "RESUME_PID=$! RUNLOG=$RUNLOG"
sleep 25
kill -0 $! 2>/dev/null && echo ALIVE || { echo DIED; tail -10 "$RUNLOG"; }
