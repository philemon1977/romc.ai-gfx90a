#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
source .env
for img in "$PYTORCH_IMAGE" "$TF_IMAGE" "$JAX_IMAGE" "$VLLM_IMAGE"; do
  echo "==== pulling $img @ $(date +%T) ===="
  docker pull "$img" || echo "PULL FAILED: $img"
done
echo "==== all pulls done @ $(date +%T) ===="
