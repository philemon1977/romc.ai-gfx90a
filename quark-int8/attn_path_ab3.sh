#!/usr/bin/env bash
# 加固版 attention 路径 A/B（前一轮被"两个 runner + 僵尸 server"搅乱，本版逐臂核验、清场彻底）
#   arm 1  attn-triton      : --attention-backend TRITON_ATTN（清单第 1 项，干净重试）
#   arm 2  spec0-splitkv    : SPEC=0 + VLLM_ROCM_SPLITKV_PA=1（长上下文真杠杆）
#   arm 3  default-restore  : 出厂默认（SPEC=5 / auto→ROCM_ATTN），**留运行**
# 每个 arm：清场 → 启动 → 就绪（有界 20 min）→ **/proc/<pid>/cmdline 核验配置** → 四口径测量
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int8w8a8attn_aiter_vllm_rocm72_mtp5_256k_8117_ornith_mi250dx8.sh
PIDF=/home/qiba/ai/logs/ornith397b-8117.pid
MODEL=/home/qiba/ai/models/ornith-ai/Ornith-1.5-397B-Quark-Int8-Attn
exec > >(tee -a "$REPO/logs/attn_path_ab3.log") 2>&1
echo "=== 加固版 A/B 开始 $(date -u +%FT%TZ) ==="

# ── 协作门（**绝不杀非本会话的服务**）──────────────────────────────────────
# 背景：本机有并行会话在调同一个模型（用户 2026-09-18 告知）。共享的 PID 文件
# `ornith397b-8117.pid` 会被双方覆盖 ⇒ 任何"按 PID 文件停服"都可能误杀对方。
# 本版规则：
#   1) 只停 **environ 里带本会话标记 AB_MARKER** 的进程（我方启动时注入该标记）；
#   2) 其余情况只**等**：等 8117 无监听、等整机无 vLLM server、等每 die VRAM ≥62 GiB；
#   3) 等待有上限（默认 40 min），等不到就放弃该臂而不是抢占。
AB_MARKER="${AB_MARKER:-qiba-attn-ab-$$}"
mine () {  # pid -> 0 表示是我方起的
  tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | grep -q "^AB_MARKER="
}
stop_mine () {
  local killed=0
  for d in /proc/[0-9]*; do
    local p=${d#/proc/}; [ -r "$d/cmdline" ] || continue
    local cl; cl=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)
    case "$cl" in
      *vllm.entrypoints.openai.api_server*) mine "$p" && { kill -TERM "$p" 2>/dev/null && { echo "  停我方 server PID $p"; killed=1; }; } ;;
    esac
  done
  rm -f "$PIDF" 2>/dev/null
  [ "$killed" = 1 ] && sleep 15
  return 0
}
foreign_servers () {  # 输出非我方的 vLLM server PID（只读，不杀）
  for d in /proc/[0-9]*; do
    local p=${d#/proc/}; [ -r "$d/cmdline" ] || continue
    local cl; cl=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)
    case "$cl" in
      *vllm.entrypoints.openai.api_server*) mine "$p" || echo "$p" ;;
    esac
  done
}
wait_quiescent () {  # 等 GPU 空出来（不抢、不杀）
  local limit=${1:-240}   # ×10s ⇒ 默认 40 min
  for i in $(seq 1 "$limit"); do
    local f; f=$(foreign_servers | tr '\n' ' ')
    local busy=0; [ -n "$f" ] && busy=1
    nc -z 127.0.0.1 8117 2>/dev/null && busy=1
    local free; free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    [ "${free:-0}" -ge 62 ] || busy=1
    if [ "$busy" = 0 ]; then echo "  ✅ GPU 已释放（min free ${free} GiB，无他人 server）"; return 0; fi
    [ $((i % 6)) = 1 ] && echo "  ⏳ 等待中（他人 server: ${f:-无}；8117: $(nc -z 127.0.0.1 8117 && echo 占用 || echo 空闲)；min free ${free:-?} GiB）"
    sleep 10
  done
  echo "  ❌ 等待超时：对方一直占着 GPU ⇒ 放弃本次启动（不抢占）"; return 1
}

verify_arm () {  # tag 期望的 cmdline 片段...
  local tag=$1; shift
  local pid=""; [ -f "$PIDF" ] && pid=$(cat "$PIDF" 2>/dev/null || true)
  # 兜底：找不到就按 cmdline 扫
  if [ -z "${pid:-}" ] || ! kill -0 "$pid" 2>/dev/null; then
    for d in /proc/[0-9]*; do
      local cl; cl=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)
      case "$cl" in *vllm.entrypoints.openai.api_server*--port\ 8117*) pid=${d#/proc/}; break;; esac
    done
  fi
  local cl; cl=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
  echo "  PID=$pid"
  echo "  实际配置: $(echo "$cl" | grep -oE '\-\-port [0-9]+|--attention-backend [A-Z_]+|--speculative-config \{[^}]*\}|--max-model-len [0-9]+|--max-num-batched-tokens [0-9]+' | tr '\n' ' ')"
  local ok=1 want
  for want in "$@"; do
    if [ "${want#!}" != "$want" ]; then          # !开头 = 断言"不应出现"
      want=${want#!}
      case "$cl" in *"$want"*) echo "  ❌ 期望**不**含「$want」但实际有 ⇒ 该臂配置未生效"; ok=0;; esac
    else
      case "$cl" in *"$want"*) ;; *) echo "  ❌ 期望含「$want」但实际没有 ⇒ 该臂配置未生效"; ok=0;; esac
    fi
  done
  [ "$ok" = 1 ] && echo "  ✅ 配置核验通过"
  return $((1-ok))
}

measure () {  # tag spec
  local tag=$1 spec=$2
  echo "--- [$tag] 短上下文 count n=5 ---"
  python3 "$REPO/measure_median.py" 8117 "$tag-count" 5 256 count 2>&1 | sed 's/^/  /'
  echo "--- [$tag] 短上下文 explain n=5 ---"
  python3 "$REPO/measure_median.py" 8117 "$tag-explain" 5 256 explain 2>&1 | sed 's/^/  /'
  echo "--- [$tag] 步时分解（count, SPEC=$spec）---"
  python3 "$REPO/step_probe.py" 8117 "$tag" "$spec" count 256 2>&1 | sed 's/^/  /'
  echo "--- [$tag] 长上下文 @16.7k ---"
  python3 "$REPO/longctx_probe.py" 8117 "$tag-16k" 16000 32 "$spec" 2>&1 | sed 's/^/  /'
  echo "--- [$tag] 长上下文 @48k ---"
  python3 "$REPO/longctx_probe.py" 8117 "$tag-48k" 48000 32 "$spec" 2>&1 | sed 's/^/  /'
}

run_arm () {  # tag spec expect1 [expect2...] -- env...
  local tag=$1 spec=$2; shift 2
  local expects=()
  while [ "$1" != "--" ]; do expects+=("$1"); shift; done
  shift
  echo; echo "############ ARM $tag  SPEC=$spec  env: $*  $(date -u +%H:%M:%S) ############"
  stop_mine
  wait_quiescent || { echo "[$tag] ⏸ 跳过（GPU 未释放）"; return 1; }
  env -u SPEC -u VLLM_ROCM_USE_AITER -u VLLM_DISABLE_COMPILE_CACHE -u AITER_CONFIG_GEMM_A8W8 \
      -u VLLM_EXTRA_ARGS -u VLLM_ROCM_SPLITKV_PA \
      PORT=8117 SPEC="$spec" AB_MARKER="$AB_MARKER" "$@" bash "$LAUNCH" || { echo "[$tag] ❌ LAUNCH FAILED"; return 1; }
  local t0; t0=$(date +%s)
  for i in $(seq 1 240); do curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 && break; sleep 5; done
  local S; S=$(ls -t /home/qiba/ai/logs/ornith397b/server-8117-*.log | head -1)
  if ! curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1; then
    echo "[$tag] ❌ NOT READY 超过 $(( $(date +%s)-t0 ))s（**挂死也要记录证据**）"
    echo "  日志 $S"
    echo "  Loading weights took 次数=$(grep -ac 'Loading weights took' "$S")  startup完整=$(grep -ac 'Application startup complete' "$S")"
    grep -aoE "(ValueError|RuntimeError|AssertionError|ImportError|CUDA out of memory): .{0,140}" "$S" | sort -u | head -3 | sed 's/^/  /'
    tail -3 "$S" | cut -c1-150 | sed 's/^/  /'
    return 1
  fi
  echo "  ✅ ready $(( $(date +%s)-t0 ))s"
  verify_arm "$tag" "${expects[@]}" || echo "  ⚠️ 该臂配置与意图不符（数字按'实际配置'解读）"
  echo "  证据：AiterInt8=$(grep -ac 'Selected AiterInt8ScaledMMLinearKernel' "$S")" \
       "TritonInt8=$(grep -ac 'Selected TritonInt8ScaledMMLinearKernel' "$S")" \
       "attn=$(grep -aoE 'Overriding with [A-Za-z_]+' "$S" | sort -u | tr '\n' ',')" \
       "notfound=$(grep -ac 'not found tuned config' "$S")" \
       "KV=$(grep -aoE 'GPU KV cache size: [0-9,]+' "$S" | head -1 | grep -oE '[0-9,]+')"
  measure "$tag" "$spec"
}

run_arm attn-triton 5 "--attention-backend TRITON_ATTN" -- VLLM_EXTRA_ARGS="--attention-backend TRITON_ATTN"
run_arm spec0-splitkv 0 "--port 8117" "!--speculative-config" -- VLLM_ROCM_SPLITKV_PA=1
run_arm default-restore 5 "--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":5}" -- 
echo; echo "=== 完成：留运行 default-restore（出厂默认配方）$(date -u +%FT%TZ) ==="
