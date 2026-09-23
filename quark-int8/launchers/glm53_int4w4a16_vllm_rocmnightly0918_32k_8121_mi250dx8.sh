#!/bin/bash
# Desc: GLM-5.3 (glm_moe_dsa) compressed-tensors INT4 W4A16 起服（8×MI250X / gfx90a）
# 挂载的补丁（与模型无关的三件 + GEMV）：
#   ⑧ _aiter_ops.py            —— 让 aiter ops 在 gfx90a 注册
#   ⑨ rocm_aiter_mla_sparse.py —— 含**自研可进图的稀疏 indexer logits 内核**（gfx90a）
#   ⑬ sparse_attn_indexer.py   —— indexer 路径门控
#   ② moe_gemv/                —— CT WNA16 MoE 的 decode 小 M GEMV（+sitecustomize）
set -euo pipefail
PORT="${PORT:-8121}"
AI_HOME="${AI_HOME:-/home/qiba/ai}"
REPO=/home/qiba/ROCm.AI
PATCH_ROOT="$AI_HOME/recipes/patches/gfx90a/ct_w4a16_dsv41_n0918/tree"
GEMV_PATCH="$AI_HOME/recipes/patches/gfx90a/ct_w4a16_dsv41_n0918/moe_gemv"
MODEL_PATH="${MODEL_PATH:-/mnt/kioxia-cm6-3t8/ai/models/ZhipuAI/GLM-5.3-CT-Int4-W4A16}"
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:nightly-0918}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.97}"
MI250_MOE_GEMV="${MI250_MOE_GEMV:-1}"
MI250_MOE_GEMV_MODULE="${MI250_MOE_GEMV_MODULE:-mi250_moe_gemv_gs}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-0}"
# 2026-09-21：ENFORCE_EAGER=0 且 capture size=0 会被 vLLM 直接断言拒绝 ——
#   "Maximum cudagraph size should be greater than or equal to 1 when using cuda graph"
# 即 eager=0 这条路在本 launcher 里从来没走通过。默认给到 MAX_NUM_SEQS：decode 批大小
# 不会超过它，capture 规模也因此有界（捕获时间与显存都可控）。
if [ "$ENFORCE_EAGER" = "0" ] && [ "$MAX_CUDAGRAPH_CAPTURE_SIZE" = "0" ]; then
  MAX_CUDAGRAPH_CAPTURE_SIZE="$MAX_NUM_SEQS"
  echo "   (ENFORCE_EAGER=0 ⇒ max_cudagraph_capture_size 自动取 $MAX_CUDAGRAPH_CAPTURE_SIZE)"
fi
LOG_DIR="${AI_HOME}/logs/glm53"; mkdir -p "$LOG_DIR"
# ★ 必须在 MOUNT 之前建好：目录不存在时 docker 会代建 root 属主目录，
#   之后调优器（普通用户）就写不进表了 —— 且整条链路静默无报错。
MOE_TUNED_DIR="${MOE_TUNED_DIR:-/home/qiba/ai/config/moe-tuned}"
mkdir -p "$MOE_TUNED_DIR" 2>/dev/null || true
LOG_FILE="${LOG_DIR}/server-${PORT}-$(date +%Y%m%d-%H%M%S).log"
PID_FILE="${AI_HOME}/logs/glm53-${PORT}.pid"

[ -d "$MODEL_PATH" ] || { echo "❌ 模型目录不在：$MODEL_PATH"; exit 1; }
[ -f "$MODEL_PATH/config.json" ] || { echo "❌ 缺 config.json"; exit 1; }
[ -f "$MODEL_PATH/model.safetensors.index.json" ] || { echo "❌ 缺 index.json"; exit 1; }
for f in _aiter_ops.py v1/attention/ops/rocm_aiter_mla_sparse.py model_executor/layers/sparse_attn_indexer.py; do
  [ -f "$PATCH_ROOT/$f" ] || { echo "❌ 补丁不在：$PATCH_ROOT/$f"; exit 1; }
done
if [ "$MI250_MOE_GEMV" != "0" ]; then
  [ -f "${GEMV_PATCH}/sitecustomize.py" ] || { echo "❌ GEMV 补丁不在"; exit 1; }
fi

MOUNT=(-v "${MODEL_PATH}:/models:ro" \
  -v "${PATCH_ROOT}/v1/attention/backends/mla/rocm_aiter_mla_sparse.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py:ro"
  -v "${PATCH_ROOT}/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:ro"
  -v "${PATCH_ROOT}/_aiter_ops.py:/usr/local/lib/python3.12/dist-packages/vllm/_aiter_ops.py:ro"
  -v "${PATCH_ROOT}/v1/attention/ops/rocm_aiter_mla_sparse.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:ro"
  -v "${PATCH_ROOT}/model_executor/layers/sparse_attn_indexer.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py:ro"
  # fastsafetensors 切块预算钩子（2026-09-20）：镜像原文件 + 我们透出 device_memory_budget /
  # max_batch_bytes 两个 env（未设时与上游逐字一致）。见 quark-int8/DCP_A_NOTES.md 装载速度节。
  -v "${PATCH_ROOT}/model_executor/model_loader/weight_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py:ro"
  -v "${GEMV_PATCH}:/patches/moe_gemv:ro"
  # MoE tile 实测表目录：vLLM 先查用户目录再查镜像目录 ⇒ 不必改镜像
  -v "${MOE_TUNED_DIR:-/home/qiba/ai/config/moe-tuned}:/moe-tuned:ro")

ENVS=(-e VLLM_ENGINE_READY_TIMEOUT_S=3600
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-256}
  # 自研分块 indexer logits 内核的取证开关（默认关）：打印每次 prefill 的
  # M/H/N 与分块数，用来核对"实际形状 vs 预算"。2026-09-20 加。
  -e MI250_INDEXER_LOGITS_DEBUG=${MI250_INDEXER_LOGITS_DEBUG:-}
  # indexer **decode** logits 路径选择（2026-09-20）：
  #   未设/"0" = 上游 torch 回退 —— 它按行主序读 SHUFFLE 页缓存，本机结果不可信；
  #   "1"      = 自研 gfx90a 内核（按 SHUFFLE 反解，已过真值三方对拍，支持 BS=16）；
  #   "auto"   = block_size>1 时自动用内核。
  -e DSV41_IDX_AITER_KERNEL=${DSV41_IDX_AITER_KERNEL:-}
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
  -e AITER_TRITON_LOG_LEVEL=ERROR
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1
  -e HF_HUB_OFFLINE=1 -e VLLM_DISABLE_COMPILE_CACHE=1
  -e VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-0}
  # QuickReduce（gfx90a C2+C3 补丁，见 hyperloom/patches-local/apply_gfx90a_quickreduce.py）：
  # ★ 三条一律“只在显式设值时才注入”（条件段在 ENVS 定义之后）——空串不等于未设置：
  #   ValueError: Invalid value '' for VLLM_ROCM_QUICK_REDUCE_QUANTIZATION.
  #   Valid options: ['FP','INT8','INT6','INT4','INT3','NONE']
  #   ⇒ worker 直接起不来（2026-09-21 07:40 实测踩过：A 臂基线 Exited(1)）。
  # 启用要同时给三条（尤其 MIN_SIZE=0，否则 decode 的 4-10KB AR 不进场）；
  # 只用 FP 无损模式，其余量化模式与 CUSTOM 在 gfx90a 上会静默算错（前人实测）。
  -e VLLM_ROCM_USE_AITER_MOE=0
  # MoE tile 实测表：vLLM 先查这个用户目录，再查镜像 configs（不必改镜像）
  -e VLLM_TUNED_CONFIG_FOLDER=${VLLM_TUNED_CONFIG_FOLDER:-/moe-tuned}
  # fastsafetensors（仅 --load-format fastsafetensors 时生效）：默认**留空 = 不设**，其它会话零影响。
  # 2026-09-20 加，起因见 quark-int8/DCP_A_NOTES.md "装载速度"节：
  #   UNIFIED_MEM=1 暂存放统一内存 ⇒ 绕开"文件级 GPU 批次 ~8 GiB + 权重 52.95 GiB 顶穿 64 GiB"
  #   ODIRECT=1     绕页缓存 ⇒ 避免 402 GB 模型对 251 GiB 内存造成的 kswapd 抖动
  -e FASTSAFETENSORS_UNIFIED_MEM=${FASTSAFETENSORS_UNIFIED_MEM:-}
  -e FASTSAFETENSORS_ODIRECT=${FASTSAFETENSORS_ODIRECT:-}
  # FST 切块：MI250_FST_DEVICE_BUDGET_MB>0 时按该预算把大文件切成小块装载 ⇒ 峰值暂存可控
  -e MI250_FST_DEVICE_BUDGET_MB=${MI250_FST_DEVICE_BUDGET_MB:-}
  -e MI250_FST_MAX_BATCH_MB=${MI250_FST_MAX_BATCH_MB:-}
  # DCP 索引/合并取证开关（默认关）：打印 q 行数 vs metadata 行数、valid vs 行长度、lse 统计
  -e MI250_DCP_DEBUG=${MI250_DCP_DEBUG:-})
# QuickReduce 三条：非空才注入。空串会被 vLLM 当非法枚举值 ⇒ worker 起不来（09-21 实测）。
# 用 if 形式而不是「[ -n ] && ...」，避免条件为假的返回码在 set -e 下把脚本带走。
if [ -n "${VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB:-}" ]; then
  ENVS+=(-e "VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB=${VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB}")
fi
if [ -n "${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-}" ]; then
  ENVS+=(-e "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION}")
fi
if [ -n "${VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16:-}" ]; then
  ENVS+=(-e "VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=${VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16}")
fi
case "$MI250_MOE_GEMV" in
  0) ;;
  *) ENVS+=(-e PYTHONPATH=/patches/moe_gemv -e MI250_MOE_GEMV=1
       -e "MI250_MOE_GEMV_MODULE=${MI250_MOE_GEMV_MODULE}"
       # v3 = scale 提出 k 循环（2026-09-20 实测 gemm1 4.1x / gemm2 4.3x；端到端 +30~100%）
       -e "MI250_MOE_GEMV_KERNEL=${MI250_MOE_GEMV_KERNEL:-}"
       # BOTH=1 时 gemm2(down) 也接管（v3 之后才有意义：v1 时代逆置换开销吃掉收益）
       -e "MI250_MOE_GEMV_BOTH=${MI250_MOE_GEMV_BOTH:-}"
       # DEBUG=1 打印 DUMP（A/C/B 真实形状 + topk_ids 样例）与 SELFCHECK 对拍
       -e "MI250_MOE_GEMV_DEBUG=${MI250_MOE_GEMV_DEBUG:-}"
       # ★ decode 稀疏注意力 split-K（2026-09-21）：S=8 实测内核 7.2x；默认 0=关闭
       -e "MI250_SPARSE_SPLITK=${MI250_SPARSE_SPLITK:-}"
       -e "MI250_SPARSE_SPLITK_MAXM=${MI250_SPARSE_SPLITK_MAXM:-}") ;;
esac

# 可选：torch profiler（设置 PROF_DIR 才挂载并开启 /start_profile）
if [ -n "${PROF_DIR:-}" ]; then
  mkdir -p "$PROF_DIR"
  MOUNT+=(-v "$PROF_DIR:/prof")
  ENVS+=(-e VLLM_TORCH_PROFILER_DIR=/prof)
  echo "   [prof] 已启用 profiler，trace 落 $PROF_DIR"
fi

ARGS=(/models
  --port "$PORT"                    # ★ 必须显式指定：vLLM 默认 8000，漏了会让探针连不上（2026-09-20 实测）
  --served-model-name glm-5.3
  --tensor-parallel-size 8
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --dtype bfloat16
  --reasoning-parser glm47 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE")
[ "$ENFORCE_EAGER" != "0" ] && ARGS+=(--enforce-eager)
# MTP 投机解码（GLM-5.3 自带 1 个 MTP 层：num_nextn_predict_layers=1；
# vLLM 用 is_mtp_layer = layer_id >= num_hidden_layers 把 layers.78 当 MTP 层）
if [ -n "${SPEC_CONFIG:-}" ]; then ARGS+=(--speculative-config "$SPEC_CONFIG"); fi
[ -n "${VLLM_EXTRA_ARGS:-}" ] && { read -r -a _x <<< "$VLLM_EXTRA_ARGS"; ARGS+=("${_x[@]}"); }

# ★★ 起服前硬门（CLAUDE.md §1：绝不抢卡、不腾地方）
#   2026-09-20 实测教训：曾有并发会话占着 8×53 GiB 跑 dsv41-ct-int4，
#   当时只有外面的盯梢脚本在等，启动器自己会照样起服 ⇒ 双方一起 OOM。
# ★ 先释放**自己**的同名容器，再判门：否则"重启自己的服务"会被自己的硬门挡住。
docker rm -f glm53-int4 > /dev/null 2>&1 || true
BUSY_GCD=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{if ($NF/1073741824 > 5) n++} END{print n+0}')
if [ "${BUSY_GCD:-0}" -gt 0 ]; then
  OCC=$(docker ps --format '{{.Names}}' | grep -vE "^glm53-int4$" | paste -sd, - || true)
  echo "❌ 有 ${BUSY_GCD} 个 GCD 显存占用 > 5 GiB ⇒ 可能被其他会话使用，拒绝起服（不腾地方）。"
  echo "   在跑的其它容器: ${OCC:-无}"
  echo "   确实要强行起：ALLOW_BUSY=1 bash 本脚本"
  [ "${ALLOW_BUSY:-0}" = "1" ] || exit 1
  echo "   ⚠️ ALLOW_BUSY=1 已指定，继续（后果自负）"
fi

echo "== GLM-5.3 INT4 W4A16 :${PORT}  len=${MAX_MODEL_LEN} util=${GPU_MEM_UTIL} eager=${ENFORCE_EAGER} GEMV=${MI250_MOE_GEMV} =="
echo "   模型 $MODEL_PATH"
# ★ 旧同名容器必须先清掉，否则 docker run 直接因名字冲突失败
#   （2026-09-20 实测：上次失败的容器留着名字 ⇒ 新容器没起来，而 docker logs 打的是旧日志，
#     表现为"引擎初始化失败"的假象）。另外 docker run -d 本身就是分离的，不能再加 &。
docker rm -f glm53-int4 > /dev/null 2>&1 || true
# CPU_SHARES（2026-09-21 加）：宿主上并发的量化转换任务会与容器抢核；A/B 期间设 256 让出 CPU
#（显存侧另由 GPU_MEM_UTIL 让出）。默认 1024 = docker 默认，行为不变。
docker run -d --name glm53-int4 --network host --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc host --shm-size 16g \
  --cpu-shares "${CPU_SHARES:-1024}" \
  "${MOUNT[@]}" "${ENVS[@]}" "$IMAGE" "${ARGS[@]}" > /dev/null
CID=""
for _i in $(seq 1 20); do
  CID=$(docker ps -q -f name=glm53-int4 || true)
  [ -n "$CID" ] && break
  sleep 1
done
if [ -z "$CID" ]; then
  echo "❌ 容器未起来（名字冲突或镜像问题）"
  docker ps -a --format "{{.Names}} {{.Status}}" | head -5
  exit 1
fi
echo "$CID" > "$PID_FILE"
docker logs -f glm53-int4 > "$LOG_FILE" 2>&1 &
echo "   ✅ 已启动 container=$CID  日志 $LOG_FILE"
echo "   停止: docker rm -f glm53-int4"