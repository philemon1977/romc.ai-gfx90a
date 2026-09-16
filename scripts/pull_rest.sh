#!/usr/bin/env bash
# 通过显式代理 registry 拉取剩余三个 ROCm 镜像并打上标准 tag
# 用法: scripts/pull_rest.sh [mirror_prefix]   (默认 dockerproxy.net)
set -uo pipefail
cd "$(dirname "$0")/.."
MIRROR=${1:-dockerproxy.net}
IMGS=(
  "rocm/tensorflow:rocm7.14.1-ubuntu24.04-py3.12-tf2.20"
  "rocm/jax:rocm7.2.4-jax0.8.2-py3.12"
  "rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909"
)
for img in "${IMGS[@]}"; do
  echo "==== pulling $MIRROR/$img @ $(date +%T)"
  if docker pull "$MIRROR/$img"; then
    docker tag "$MIRROR/$img" "$img"
    echo "==== done: $img"
  else
    echo "==== FAILED: $img"
  fi
done
echo "==== pull_rest finished @ $(date +%T)"
