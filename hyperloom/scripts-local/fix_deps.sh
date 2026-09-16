#!/usr/bin/env bash
# 容器内运行: 安装 hyperloom 1.1.0 的 [llm,forge] 运行时依赖
set -uo pipefail
P=/opt/envs/vllm/bin/python
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  "$P" -m pip install --break-system-packages "hyperloom-inference-optimizer[llm,forge]==1.1.0" \
  2>&1 | tail -5
"$P" -c "import claude_agent_sdk; print('claude_agent_sdk OK')"
