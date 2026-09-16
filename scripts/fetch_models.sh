#!/usr/bin/env bash
# =============================================================================
# 下载冒烟用 tiny 模型到 ./models/（模型权重不入库，见 DEPENDENCIES.md）
#
# 本机网络经验（README）：HF 直连被 SSL 截断，走 ModelScope。
# 用法： scripts/fetch_models.sh [model_id] [target_dir]
#        默认 Qwen/Qwen2.5-0.5B-Instruct -> ./models
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_ID="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
DEST="${2:-$ROOT/models}"
BASE="https://modelscope.cn/models/${MODEL_ID}/resolve/master"

FILES=(
  config.json
  generation_config.json
  tokenizer_config.json
  tokenizer.json
  vocab.json
  merges.txt
  model.safetensors
)

mkdir -p "$DEST"
echo "目标：$DEST   来源：$BASE"
fail=0
for f in "${FILES[@]}"; do
  if [[ -s "$DEST/$f" ]]; then
    echo "   已存在 $f（$(du -h "$DEST/$f" | cut -f1)），跳过"
    continue
  fi
  echo "   下载 $f ..."
  # 断点续传；失败不中断其余文件（部分模型无 merges.txt 等）
  curl -fL --retry 3 --retry-delay 2 -C - -o "$DEST/$f" "$BASE/$f" \
    || { echo "   !! 失败：$f（该模型可能不含此文件）" >&2; fail=1; }
done

[[ -s "$DEST/config.json" && -s "$DEST/model.safetensors" ]] \
  || { echo "必需文件缺失，models/ 不可用" >&2; exit 1; }

echo
echo "OK: $DEST 就绪。启动服务："
echo "  docker-compose up -d vllm          # 默认 vllm serve /models"
echo "  或 docker-compose run --rm -e HIP_VISIBLE_DEVICES=1 vllm \\"
echo "       vllm serve /models --served-model-name tiny-qwen --max-model-len 4096"
exit $fail
