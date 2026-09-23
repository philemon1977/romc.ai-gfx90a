#!/usr/bin/env bash
# 准备 Hyperloom 用的容器：让 hyperloom-local 能起 GLM-5.3-Flash Quark-INT8
#
# 为什么必须动容器（2026-09-21 取证，全部纯 CPU/容器操作，不占卡）：
#   1. 模型不可见：hyperloom-local 只挂了 /home/qiba/ROCm.AI(rw) 与
#      /mnt/kioxia-cm6-3t8/ai/models(ro)；本权重在 /mnt/stripe-3mix-3t2/models，
#      容器里 ls 不到 => Magpie 起的 vllm serve 直接找不到模型。
#   2. 缺 8 号补丁：容器内 vllm 0.3.1.dev85+gdee37d891 的
#      models/glm5next/amd/sparse_indexer.py 与 model_executor/layers/mhc.py
#      与 -0918 底座逐字节相同（size+diff 已对拍）=> glm5_next 一定撞
#      "Sparse attention indexer ROCm path is only supported on AITER."，
#      且 tilelang mHC 在 gfx90a 上算错 layer_input（见根因报告）。
#   3. 不能直接从底座镜像 build：容器可写层里有 Magpie 的 MI250X runner 注册
#      （apply_mi250x_runner.py 在容器内跑过），重建会丢 => 用 commit 保住。
#
# 可回滚：旧容器只 rename+stop，不 rm。
set -euo pipefail

OLD=hyperloom-local
IMG_OUT=rocm-ai/vllm:glm53-int4-hl-fl1
VPKG=/usr/local/lib/python3.12/dist-packages/vllm
PT=${AI_HOME}/recipes/patches/vllm/vllm-openai-rocm-nightly-0918/core/tree
STRIPE=/mnt/stripe-3mix-3t2/models
KIOXIA=/mnt/kioxia-cm6-3t8/ai/models
WS=/home/qiba/ROCm.AI

FILES=(
  "models/glm5next/amd/sparse_indexer.py"
  "model_executor/layers/mhc.py"
)

echo "== 0) 前置检查 =="
docker inspect "$OLD" >/dev/null
for f in "${FILES[@]}"; do
  [ -f "$PT/$f" ] || { echo "MISSING patch file: $PT/$f"; exit 1; }
done
[ -d "$STRIPE" ] || { echo "MISSING model dir on host: $STRIPE"; exit 1; }
docker image inspect "$IMG_OUT" >/dev/null 2>&1 && echo "NOTE: target image exists, will overwrite" || true

echo "== 1) 备份容器内原件（带 .orig-0918 后缀，只备一次） =="
for f in "${FILES[@]}"; do
  docker exec "$OLD" bash -c "cp -n $VPKG/$f $VPKG/$f.orig-0918 || true"
done

echo "== 2) 落补丁 =="
for f in "${FILES[@]}"; do
  docker cp "$PT/$f" "$OLD:$VPKG/$f"
done
for f in "${FILES[@]}"; do
  printf 'marker %s : ' "$f"
  docker exec "$OLD" bash -lc "grep -c 'gfx90a-host patch' $VPKG/$f"
done
echo "-- py_compile（只编这两个文件，不导入 vllm）--"
docker exec "$OLD" bash -lc "python3 -m py_compile $VPKG/models/glm5next/amd/sparse_indexer.py $VPKG/model_executor/layers/mhc.py && echo PYCOMPILE_OK"
echo "-- 闸门实读 --"
docker exec "$OLD" bash -lc "grep -n 'if on_gfx90a():' $VPKG/model_executor/layers/mhc.py | head -3"
docker exec "$OLD" bash -lc "grep -n 'or on_gfx90a()' $VPKG/models/glm5next/amd/sparse_indexer.py | head -3"

echo "== 3) commit 成新镜像（保住 Magpie 的 mi250x runner 注册） =="
docker commit "$OLD" "$IMG_OUT"
docker image inspect "$IMG_OUT" --format 'IMGID={{.Id}} ENTRY={{json .Config.Entrypoint}} CMD={{json .Config.Cmd}}'

echo "== 4) rename + stop 旧容器 =="
docker rename "$OLD" hyperloom-local-prev
docker stop hyperloom-local-prev

echo "== 5) 用新镜像起同名容器（多挂一个模型盘，只读） =="
docker run -d --name hyperloom-local \
  --entrypoint tail \
  --device /dev/kfd --device /dev/dri \
  --group-add video \
  --shm-size 64g \
  -v "$WS:$WS" \
  -v "$KIOXIA:$KIOXIA:ro" \
  -v "$STRIPE:$STRIPE:ro" \
  "$IMG_OUT" -f /dev/null

echo "== 6) 新容器自检 =="
docker inspect hyperloom-local --format 'IMG={{.Config.Image}}'
docker exec hyperloom-local bash -lc 'ls '"$STRIPE"'/ZhipuAI/GLM-5.3-Flash-Quark-Int8 | head -8'
docker exec hyperloom-local bash -lc "grep -c 'gfx90a-host patch' $VPKG/models/glm5next/amd/sparse_indexer.py $VPKG/model_executor/layers/mhc.py"
docker exec hyperloom-local bash -lc 'ls -l /dev/kfd | head -2; id'
echo "-- .env 的 HYPERLOOM_IMAGE 改成新 tag（留备份）--"
docker exec hyperloom-local bash -lc 'E=/home/qiba/ROCm.AI/hyperloom/.env; cp -n $E $E.bak-fl1 2>/dev/null || true; sed -i "s#^HYPERLOOM_IMAGE=.*#HYPERLOOM_IMAGE=rocm-ai/vllm:glm53-int4-hl-fl1#" $E; grep -E "^(HYPERLOOM_IMAGE|HYPERLOOM_RUN_MODE|USER_DATA_PATH|HYPERLOOM_CONTAINER_NAME)=" $E'

echo "ALL_DONE"
