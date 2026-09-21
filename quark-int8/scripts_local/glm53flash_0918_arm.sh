#!/bin/bash
# Desc: GLM-5.3-Flash-Quark-Int8 在**镜像** rocm-ai/vllm:glm53-int4-gfx90a-0918 上起服（对照 8127 的宿主 editable 树）
# 两道门叠着（2026-09-21 源码级定位；门 A 方案未起服即被否证）：
#   门一（起不来）vllm/models/glm5next/amd/sparse_indexer.py:684
#       "if not rocm_aiter_ops.is_enabled(): raise Sparse attention indexer ROCm path..."
#       glm5_next 是新架构包，自带 amd/nvidia/common 三套分派；ROCm 取 amd 那棵，
#       而补丁树里原本一件 models/glm5next/** 都没有 ⇒ 第 8 件补丁放行（IDX_PATCH=1）。
#       ⚠️ 不要用 VLLM_ROCM_USE_AITER=1 过这道门：打开总闸会让实现体
#          (v1/attention/ops/rocm_aiter_mla_sparse.py:841) 先抢进 aiter/deepgemm 分支，
#          那条内核在 gfx90a 上编译不过；且子闸 MLA/MHA/LINEAR/RMSNORM 默认 True，
#          会连带打开 gfx942/950 专属 CK。保持 AITER=0 才停在 gfx90a Triton 分派上。
#   门二（数值坏）该镜像的 mhc.py 不含 gfx90a 排除且容器内装了 tilelang（本次日志已证：
#       08:17:58 TileLang 编 hc_prenorm / mhc_pre_*）⇒ MHC_PATCH=1 挂已修那份
# 端口 8128（8127 是另一会话在途臂的把手，不复用）；容器名 glm53flash-0918
set -euo pipefail

PORT="${PORT:-8128}"
NAME="glm53flash-0918"
AI_HOME="${AI_HOME:-/home/qiba/ai}"
REPO=/home/qiba/ROCm.AI
IMAGE="${IMAGE:-rocm-ai/vllm:glm53-int4-gfx90a-0918}"
MODEL_PATH="${MODEL_PATH:-/mnt/stripe-3mix-3t2/models/ZhipuAI/GLM-5.3-Flash-Quark-Int8}"
PATCH_ROOT="$AI_HOME/patches/gfx90a/ct_w4a16_dsv41_n0918/tree"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"          # kpool 硬门：index_kpool=4 ⇒ 必须 128 的倍数
MHC_PATCH="${MHC_PATCH:-0}"              # 1 = 挂载已修 mhc.py（门二）
IDX_PATCH="${IDX_PATCH:-1}"              # 1 = 挂载放行 gfx90a 的 glm5next AMD indexer（门一）
                                         #   0 = 镜像原样 ⇒ 权重装完后的第一次 forward 必抛

LOG_DIR="$AI_HOME/logs/glm53flash-0918"; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/server-$PORT-$(date +%Y%m%d-%H%M%S).log"
HANDLE="$LOG_DIR/$PORT.cid"

# ── 前置实物检查 ───────────────────────────────────────────────
[ -f "$MODEL_PATH/config.json" ] || { echo "❌ 模型不在：$MODEL_PATH"; exit 1; }
[ -f "$MODEL_PATH/model.safetensors.index.json" ] || { echo '❌ 缺 index.json'; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "❌ 镜像不在：$IMAGE"; exit 1; }
if [ "$MHC_PATCH" = "1" ]; then
  M="$PATCH_ROOT/model_executor/layers/mhc.py"
  [ -f "$M" ] || { echo "❌ 补丁 mhc.py 不在：$M"; exit 1; }
  grep -q 'if on_gfx90a():' "$M" || { echo '❌ 补丁 mhc.py 里没有 gfx90a 排除（fail-closed）'; exit 1; }
fi
if [ "$IDX_PATCH" = "1" ]; then
  X="$PATCH_ROOT/models/glm5next/amd/sparse_indexer.py"
  [ -f "$X" ] || { echo "❌ 第 8 件补丁不在：$X"; exit 1; }
  grep -q 'gfx90a-host patch: glm5next' "$X" || { echo '❌ indexer 补丁缺标记（fail-closed）'; exit 1; }
  grep -q 'or on_gfx90a()' "$X" || { echo '❌ indexer 补丁没放行 gfx90a（fail-closed）'; exit 1; }
  python3 -m py_compile "$X" || { echo "❌ indexer 补丁语法不过（fail-closed）"; exit 1; }
fi

# ── 起服前的门（标准件；不抢卡不腾地方）──────────────────────
source "$REPO/quark-int8/gpu_gate.sh"
gate "$PORT" || { echo '❌ 门未过 ⇒ 不起服'; exit 1; }

# 先清自己的同名容器，再判硬门（否则重启自己会被自己挡住）
docker rm -f "$NAME" >/dev/null 2>&1 || true
BUSY_GCD=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{if ($NF/1073741824 > 5) n++} END{print n+0}')
if [ "${BUSY_GCD:-0}" -gt 0 ] && [ "${ALLOW_BUSY:-0}" != "1" ]; then
  echo "❌ ${BUSY_GCD} 个 GCD 占用 >5 GiB ⇒ 别的会话在用，拒绝起服"; exit 1
fi

MOUNT=(-v "$MODEL_PATH:/models:ro"
       -v "$LOG_DIR:/logs")
[ "$MHC_PATCH" = "1" ] && MOUNT+=(-v "$PATCH_ROOT/model_executor/layers/mhc.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mhc.py:ro")
[ "$IDX_PATCH" = "1" ] && MOUNT+=(-v "$PATCH_ROOT/models/glm5next/amd/sparse_indexer.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/amd/sparse_indexer.py:ro")

ENVS=(-e VLLM_ENGINE_READY_TIMEOUT_S=3600
      -e HF_HUB_OFFLINE=1
      -e VLLM_ROCM_USE_AITER=0 -e VLLM_ROCM_USE_AITER_MOE=0
      # 我们的 GEMV 钩的是 CT W4A16 专家路径，int8 臂上无关 ⇒ 关掉少一个变量
      -e MI250_MOE_GEMV=0
      -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
      -e AITER_TRITON_LOG_LEVEL=ERROR
      -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256)

ARGS=(/models
      --port "$PORT"
      --served-model-name glm53flash-int8
      --tensor-parallel-size 8
      --dtype bfloat16
      --gpu-memory-utilization "$GPU_MEM_UTIL"
      --max-model-len "$MAX_MODEL_LEN"
      --max-num-seqs "$MAX_NUM_SEQS"
      --block-size "$BLOCK_SIZE"
      --kv-cache-dtype bfloat16
      --language-model-only
      --safetensors-load-strategy eager
      --enforce-eager)
[ -n "${EXTRA_ARGS:-}" ] && { read -r -a _x <<< "$EXTRA_ARGS"; ARGS+=("${_x[@]}"); }

echo "== GLM-5.3-Flash Quark-INT8 :$PORT  镜像=$IMAGE  IDX_PATCH=$IDX_PATCH MHC_PATCH=$MHC_PATCH  len=$MAX_MODEL_LEN util=$GPU_MEM_UTIL blk=$BLOCK_SIZE eager=1 AITER=0 =="
docker run -d --name "$NAME" --network host --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc host --shm-size 16g \
  "${MOUNT[@]}" "${ENVS[@]}" "$IMAGE" "${ARGS[@]}" > /dev/null

CID=""
for _i in $(seq 1 20); do CID=$(docker ps -q -f "name=^${NAME}$" || true); [ -n "$CID" ] && break; sleep 1; done
[ -n "$CID" ] || { echo '❌ 容器没起来'; docker ps -a --format '{{.Names}} {{.Status}}' | head -5; exit 1; }
echo "$CID" > "$HANDLE"
docker logs -f "$NAME" > "$LOG_FILE" 2>&1 &
echo "   ✅ container=$CID  日志 $LOG_FILE"
echo "   就绪签名: 'Application startup complete'（int8 308 GiB，装载实测 354–722 s）"
echo "   验尺子:   python3 $AI_HOME/tools/probe_nll.py --url http://127.0.0.1:$PORT/v1 --model glm53flash-int8"
echo "   停服:     docker rm -f $NAME   （PID 文件里是容器 ID，别用 kill -TERM -\$(cat …)）"