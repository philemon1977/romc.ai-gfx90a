#!/usr/bin/env bash
###############################################################################
# Magpie Generic vLLM Benchmark Script for MI250X (gfx90a / CDNA2)
#
# 由来：本文件是从 Magpie 自带的 vllm_mi300x.sh 派生的 MI250X 版本（2026-09-21）。
# 与 mi300x 版的**必要差异**（每条都有实测依据，见
# hyperloom/reports/models/glm53-int4/hyperloom-mi250x-support-plan.md）：
#   1. AITER 默认关闭：gfx90a 上 AITER 的 MoE 路径不可用（本仓多处实测），
#      而 mi300x 版默认 VLLM_ROCM_USE_AITER=1。
#   2. 本机自研 gfx90a 补丁的环境（全部用 :- 默认值，环境变量可覆盖）：
#      MI250_MOE_GEMV（MoE decode GEMV，v3 内核）、DSV41_IDX_AITER_KERNEL（DCP 下
#      indexer decode logits 必须走自研内核，否则 top-K 选错）、fastsafetensors 装载。
#   3. --gpu-memory-utilization 改为读 GPU_MEMORY_UTILIZATION（mi300x 版把它写死 0.95）。
#   4. HF_HUB_OFFLINE 默认 1：本机模型都是本地路径，避免 hf download 卡在网络上。
#
# Phases (via MAGPIE_RUN_PHASE): all | server | client (default all).
# Server-only writes PID to MAGPIE_SERVER_PID_FILE then disowns and exits.
#
# Remote server (BENCHMARK_BASE_URL): when set, the client phase points
# benchmark_serving at an external vLLM-compatible HTTP endpoint.

source "$(dirname "$0")/benchmark_lib.sh"
source "$(dirname "$0")/server_cleanup.sh"
# shellcheck source=magpie_bench_remote_compat.sh
[[ -f "$(dirname "$0")/magpie_bench_remote_compat.sh" ]] && source "$(dirname "$0")/magpie_bench_remote_compat.sh"

export RUNNER_TYPE="${RUNNER_TYPE:-mi250x}"

PHASE="${MAGPIE_RUN_PHASE:-all}"
case "$PHASE" in
  all|server|client) ;;
  *) echo "ERROR: Invalid MAGPIE_RUN_PHASE='$PHASE'. Must be all|server|client." >&2; exit 2 ;;
esac

if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
  if [[ "$PHASE" != "client" ]]; then
    echo "[vllm_mi250x] BENCHMARK_BASE_URL set; forcing PHASE=client (was $PHASE)"
    PHASE=client
  fi
fi

if [[ "$PHASE" == "server" || "$PHASE" == "all" ]]; then
  check_env_vars MODEL TP
fi
if [[ "$PHASE" == "client" || "$PHASE" == "all" ]]; then
  check_env_vars MODEL CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME
fi

MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$PHASE" != "client" ]]; then
  hf download "$MODEL" 2>/dev/null || true
fi

# MI250X / gfx90a：MEC 固件旧时禁用 RCCL scratch reclaim（沿用 mi300x 版的探测方式）
version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$version" == "" || $version -lt 177 ]]; then
  export HSA_NO_SCRATCH_RECLAIM=1
fi

# ROCR_VISIBLE_DEVICES already re-indexes visible GPUs to 0..N-1, so HIP
# must use the logical range, not the original physical ids.
if [ -n "$ROCR_VISIBLE_DEVICES" ] && [ -z "$HIP_VISIBLE_DEVICES" ]; then
    n=$(echo "$ROCR_VISIBLE_DEVICES" | awk -F, '{print NF}')
    export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((n-1)))
fi

# 差异 1：gfx90a 上 AITER 不可用（尤其 MoE 路径），默认关闭
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-0}"
export VLLM_ROCM_USE_AITER_MOE="${VLLM_ROCM_USE_AITER_MOE:-0}"

# 差异 2：本机自研 gfx90a 补丁（默认开，环境可覆盖）
export MI250_MOE_GEMV="${MI250_MOE_GEMV:-1}"
export MI250_MOE_GEMV_MODULE="${MI250_MOE_GEMV_MODULE:-mi250_moe_gemv_gs}"
export DSV41_IDX_AITER_KERNEL="${DSV41_IDX_AITER_KERNEL:-1}"
export FASTSAFETENSORS_ODIRECT="${FASTSAFETENSORS_ODIRECT:-1}"
export MI250_FST_MAX_BATCH_MB="${MI250_FST_MAX_BATCH_MB:-2560}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/patches/moe_gemv:${PYTHONPATH:-}"

# 差异 4：本机模型都是本地路径，默认离线，避免 hf download 卡网络
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

WORKSPACE_DIR=${RESULT_DIR:-/workspace}
SERVER_LOG=${SERVER_LOG:-$WORKSPACE_DIR/server.log}
PORT=${PORT:-8888}

PROFILER_ARGS=()
if [[ "${PROFILE:-}" == "1" ]]; then
  TRACE_DIR="${VLLM_TORCH_PROFILER_DIR:-$WORKSPACE_DIR/torch_trace}"
  mkdir -p "$TRACE_DIR"
  PROFILER_ARGS+=(--profiler-config.profiler torch)
  PROFILER_ARGS+=(--profiler-config.torch_profiler_dir "$TRACE_DIR")
  PROFILER_ARGS+=(--profiler-config.torch_profiler_record_shapes True)
  PROFILER_ARGS+=(--profiler-config.torch_profiler_with_memory True)
  PROFILER_ARGS+=(--profiler-config.torch_profiler_with_flops True)
  PROFILER_ARGS+=(--profiler-config.torch_profiler_use_gzip True)
fi

set -x
if [[ "$PHASE" == "server" || "$PHASE" == "all" ]]; then
  setsid vllm serve $MODEL --port $PORT \
    --tensor-parallel-size=$TP \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len $MAX_MODEL_LEN \
    --trust-remote-code \
    "${PROFILER_ARGS[@]}" \
    $EXTRA_VLLM_ARGS > $SERVER_LOG 2>&1 &

  SERVER_PID=$!
  if [[ "$PHASE" == "all" ]]; then
    trap 'magpie_stop_benchmark_server_stack "$SERVER_PID"' EXIT INT TERM
  fi

  wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

  if [[ "$PHASE" == "server" ]]; then
    if [[ -z "${MAGPIE_SERVER_PID_FILE:-}" ]]; then
      echo "ERROR: MAGPIE_SERVER_PID_FILE must be set for MAGPIE_RUN_PHASE=server" >&2
      kill -TERM "-$SERVER_PID" 2>/dev/null || true
      exit 3
    fi
    printf '%s\n' "$SERVER_PID" > "$MAGPIE_SERVER_PID_FILE"
    disown "$SERVER_PID" 2>/dev/null || true
    exit 0
  fi
fi

SERVER_MONITOR_ARGS=()
if [[ -n "${SERVER_PID:-}" ]]; then
  SERVER_MONITOR_ARGS+=(--server-pid "$SERVER_PID")
fi

if [[ "$PHASE" == "client" || "$PHASE" == "all" ]]; then
  if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
    SERVER_MONITOR_ARGS=()
    magpie_run_benchmark_serving_remote_direct trust || exit $?
  else
    run_benchmark_serving \
        --model "$MODEL" \
        --port "$PORT" \
        --backend vllm \
        --input-len "$ISL" \
        --output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts ${NUM_PROMPTS:-$(( $CONC * 10 ))} \
        --max-concurrency "$CONC" \
        --result-filename "$RESULT_FILENAME" \
        --result-dir "$WORKSPACE_DIR/" \
        "${SERVER_MONITOR_ARGS[@]}" \
        --trust-remote-code || exit $?
  fi
fi

if [[ "$PHASE" != "server" && "${RUN_EVAL}" = "true" ]]; then
    if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
        if declare -F magpie_run_eval_remote_direct &>/dev/null; then
            magpie_run_eval_remote_direct || exit $?
        else
            echo "[vllm_mi250x] RUN_EVAL=true with BENCHMARK_BASE_URL but magpie_run_eval_remote_direct shim not available; skipping eval (results gate will see accuracy=None)."
        fi
    else
        run_eval --framework lm-eval --port "$PORT" || exit $?
        append_lm_eval_summary
    fi
fi
set +x
