#!/usr/bin/env bash
# 冒烟测试: 在每个 ROCm 环境容器内运行 GPU 检测与最小计算
set -uo pipefail
cd "$(dirname "$0")/.."
source .env

fail=0
run() {
  local name=$1 img=$2 script=$3
  echo "══════════ $name ($img) ══════════"
  docker run --rm \
    --device /dev/kfd --device /dev/dri \
    --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/card0)" \
    --security-opt seccomp=unconfined \
    --ipc=host --shm-size 16g \
    -v "$PWD/tests:/tests:ro" \
    -v "$PWD/models:/models:ro" \
    "$img" python /tests/"$script" || fail=1
}

sel=${1:-all}
[[ $sel == all || $sel == pytorch    ]] && run PyTorch    "$PYTORCH_IMAGE" test_pytorch.py
[[ $sel == all || $sel == tensorflow ]] && run TensorFlow "$TF_IMAGE"      test_tensorflow.py
[[ $sel == all || $sel == jax        ]] && run JAX        "$JAX_IMAGE"     test_jax.py
[[ $sel == all || $sel == vllm       ]] && run vLLM       "$VLLM_IMAGE"    test_vllm.py
exit $fail
