#!/bin/bash
# DCP 起服守卫 + 配方（2026-09-20）
# 为什么需要它：上游 launcher 第 91/107 行**无条件** `docker rm -f glm53-int4`，
# 且端口写死 8121 —— 两个会话同时起服会互相顶掉（CLAUDE.md §1 记载过同类事故，
# 2026-09-20 17:38 本线程也真实撞过一次）。这里把"起服前硬门"机制化，而不是靠自觉。
#
# 用法：
#   DCP_SIZE=2 MAX_MODEL_LEN=131072 bash scripts_local/glm_dcp_boot.sh     # ① DCP=2 + 128K
#   DCP_SIZE=8 MAX_MODEL_LEN=1048576 bash scripts_local/glm_dcp_boot.sh    # ② DCP=8 + 1M
#   FORCE=1 ...  仅当确认上面那个容器确实是自己的陈旧残留时才用。
set -uo pipefail

PORT="${PORT:-8121}"
NAME="glm53-int4"
DCP_SIZE="${DCP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
AI_HOME="${AI_HOME:-/home/qiba/ai}"
LAUNCHER="$AI_HOME/models/ZhipuAI/launcher/glm53_int4w4a16_vllm_rocmnightly0918_32k_8121_mi250dx8.sh"
FREE_GIB_MIN="${FREE_GIB_MIN:-8}"   # 每个 GCD 允许的最大已用显存（GiB），超出即认为有人在用卡

fail() { echo "❌ $*" >&2; exit 1; }

# ---- 门 1：容器名占用（launcher 会 rm -f 同名容器，必须先确认不是别人的）----
if docker ps -a --format "{{.Names}}" | grep -qx "$NAME"; then
  if [ "${FORCE:-0}" != "1" ]; then
    echo "⚠️  已存在容器 $NAME："
    docker ps -a --filter "name=^${NAME}$" --format "   {{.ID}} {{.Status}} {{.CreatedAt}}"
    fail "拒绝起服：launcher 会 docker rm -f $NAME 顶掉它。确认是自己的残留再 FORCE=1。"
  fi
  echo "⚠️  FORCE=1：按自己的残留处理 $NAME"
fi

# ---- 门 2：端口占用 ----
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  fail "拒绝起服：端口 ${PORT} 已被监听（可能是别的会话的服务，别动它）。"
fi

# ---- 门 3：显存空闲（8 卡全要）----
busy=""
while read -r gpu used; do
  used_gib=$(( used / 1073741824 ))
  if [ "$used_gib" -gt "$FREE_GIB_MIN" ]; then busy="$busy $gpu(${used_gib}GiB)"; fi
done < <(rocm-smi --showmeminfo vram 2>/dev/null | grep -a "VRAM Total Used" | \
         sed -E "s/GPU\[([0-9]+)\].*: ([0-9]+).*/\1 \2/")
if [ -n "$busy" ]; then
  fail "拒绝起服：这些 GCD 仍被占用 ⇒$busy（等对方释放，别抢卡）。"
fi

# ---- 门 4：文件与前置补丁自检 ----
[ -f "$LAUNCHER" ] || fail "launcher 不在：$LAUNCHER"
echo "== DCP 起服：size=$DCP_SIZE max_model_len=$MAX_MODEL_LEN port=$PORT =="
echo "   上游补丁自检（起服前确认三件已上树，否则会静默跑成 DCP=1 或直接报错）："
if [ "${SKIP_PATCH_CHECK:-0}" = "1" ]; then
  echo "   （SKIP_PATCH_CHECK=1：跳过补丁自检，仅用于非 DCP 裸测，如 1M KV 容量测量）"
fi
for f in "v1/attention/ops/rocm_aiter_mla_sparse.py:WRITE_LSE" \
         "v1/attention/backends/mla/rocm_aiter_mla_sparse.py:supports_dcp = True" \
         "model_executor/layers/sparse_attn_indexer.py:_merge_dcp_topk_global_gfx90a"; do
  if [ "${SKIP_PATCH_CHECK:-0}" = "1" ]; then break; fi
  path="$AI_HOME/recipes/patches/vllm/vllm-openai-rocm-nightly-0918/core/tree/${f%%:*}"
  want="${f##*:}"
  grep -q "$want" "$path" || fail "补丁缺失：$path 里找不到 [$want]（先按 dcp_patches/README.md 上树）"
  echo "   ✓ ${f%%:*} 含 [$want]"
done

# ---- 起服（DCP 参数经 launcher 的 VLLM_EXTRA_ARGS 钩子透传，不改 launcher）----
# ★ DCP 必备：indexer 的 decode logits 路径必须走自研 gfx90a 内核。
#   2026-09-20 实测：DCP 下 KV block_size 变成 16 ⇒ 默认的"上游 torch 回退"在 SHUFFLE 页缓存上
#   按行主序读、结果不可信（我们自己 ops 里就有这条 ERROR 日志），表现为 top-K 选错、
#   输出乱码（事实召回 0/6，而 DCP=1 时 6/6）。已在 launcher 有透传钩子，这里默认打开。
export DSV41_IDX_AITER_KERNEL="${DSV41_IDX_AITER_KERNEL:-1}"

# ★ 装载提速（2026-09-20 实测 7×）：fastsafetensors + 切块 + O_DIRECT。
#   实测 Model loading took **108.66 s**（mmap 是 666.6 / 769.1 s），起服总耗时 121 s。
#   三个条件缺一不可（见 DCP_A_NOTES.md "装载速度"节）：
#     ① --load-format fastsafetensors（否则走 mmap，碰不到这条路径）；
#     ② MI250_FST_MAX_BATCH_MB ≥ 最大单张量（本模型 embed/lm_head = 1.77 GiB，取 2560 MiB）；
#        注意别用 device_memory_budget —— 那个是"整个模型的设备预算"，会直接 BudgetInfeasible；
#     ③ FASTSAFETENSORS_ODIRECT=1（绕页缓存；也顺手消掉 kswapd 抖动）。
#   默认打开；要回退 mmap：FAST_LOAD=0 bash 本脚本。
if [ "${FAST_LOAD:-1}" != "0" ]; then
  export MI250_FST_MAX_BATCH_MB="${MI250_FST_MAX_BATCH_MB:-2560}"
  export FASTSAFETENSORS_ODIRECT="${FASTSAFETENSORS_ODIRECT:-1}"
  export VLLM_EXTRA_ARGS="--load-format fastsafetensors ${VLLM_EXTRA_ARGS:-}"
fi

# DCP_SIZE=0 ⇒ 不开 DCP（② 前置的"1M 裸测"：只读 Available KV cache memory）
if [ "$DCP_SIZE" != "0" ]; then
  export VLLM_EXTRA_ARGS="--decode-context-parallel-size ${DCP_SIZE} ${VLLM_EXTRA_ARGS:-}"
fi
export MAX_MODEL_LEN PORT
echo "   VLLM_EXTRA_ARGS=$VLLM_EXTRA_ARGS"
echo "   起服后必须核对：日志里的 GPU KV cache size 与 dcp 相关生效信息（见 DCP_PORT_PLAN.md 验收②）"
exec bash "$LAUNCHER"
