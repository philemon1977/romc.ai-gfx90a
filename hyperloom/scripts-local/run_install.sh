#!/usr/bin/env bash
# 容器内运行: Hyperloom 运行时安装器 (IR-2)
set -uo pipefail
export REPO_ROOT="/home/qiba/ROCm.AI/hyperloom"
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
_dotenv_prev="$(export -p | grep -v -e '=""$' -e "=''\$")"
set -a; . "${REPO_ROOT}/.env"; set +a
eval "$_dotenv_prev"
unset _dotenv_prev
export USER_DATA_PATH="${USER_DATA_PATH:?USER_DATA_PATH missing}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
ulimit -Sn 65536 || true
bash "${REPO_ROOT}/hyperloom/inference_optimizer/assets/install.sh" 2>&1 | tail -30
echo "INSTALL_RC=${PIPESTATUS[0]}"
. "$USER_DATA_PATH/runtime/kernel-agent.env.sh" && echo "KERNEL_AGENT_ENV SOURCED OK"
