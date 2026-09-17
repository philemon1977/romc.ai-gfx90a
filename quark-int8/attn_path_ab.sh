#!/usr/bin/env bash
# attention 路径三臂 A/B（每臂一次冷启；每臂都量"短上下文 + 长上下文"两个口径）
#   B  attn-triton : --attention-backend TRITON_ATTN                （清单第 1 项）
#   D  spec0-splitkv: SPEC=0 + VLLM_ROCM_SPLITKV_PA=1               （长上下文真杠杆：
#       本会话已证 SPEC>0 时 splitKV 补丁的门 max_query_len==1 不成立 ⇒ 一步都不接管）
#   C  attn-rocm   : --attention-backend ROCM_ATTN（=默认 auto 的落点，兼作对照与收尾态）
# 口径：短=count/explain（固定 prompt+丢首请求+中位数）；长=16.7k/48k 上下文（TTFT与decode分开）
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int8w8a8attn_aiter_vllm_rocm72_mtp5_256k_8117_ornith_mi250dx8.sh
PIDF=/home/qiba/ai/logs/ornith397b-8117.pid
exec > >(tee -a "$REPO/logs/attn_path_ab.log") 2>&1
echo "=== attention 路径 A/B  $(date -u +%FT%TZ) ==="

stop_srv () {
  if [ -f "$PIDF" ]; then
    local p; p=$(cat "$PIDF" 2>/dev/null || true)
    [ -n "${p:-}" ] && kill -0 "$p" 2>/dev/null && { echo "  停服 kill -TERM -$p"; kill -TERM -"$p" 2>/dev/null; }
  fi
  rm -f "$PIDF"
  for _ in $(seq 1 90); do
    nc -z 127.0.0.1 8117 2>/dev/null && { sleep 5; continue; }
    free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    [ "${free:-0}" -ge 62 ] && { echo "  已停稳（min free ${free} GiB）"; return 0; }
    sleep 5
  done
  echo "  ⚠️ 显存未回收完"; return 1
}

measure () {  # tag spec
  local tag=$1 spec=$2
  echo "--- [$tag] 短上下文 count n=5 ---"
  python3 "$REPO/measure_median.py" 8117 "$tag-count" 5 256 count 2>&1 | sed 's/^/  /'
  echo "--- [$tag] 短上下文 explain n=5 ---"
  python3 "$REPO/measure_median.py" 8117 "$tag-explain" 5 256 explain 2>&1 | sed 's/^/  /'
  echo "--- [$tag] 步时分解（短上下文 count）---"
  python3 "$REPO/step_probe.py" 8117 "$tag" "$spec" count 256 2>&1 | sed 's/^/  /'
  echo "--- [$tag] 长上下文 @16.7k ---"
  python3 "$REPO/longctx_probe.py" 8117 "$tag-16k" 16000 32 "$spec" 2>&1 | sed 's/^/  /'
  echo "--- [$tag] 长上下文 @48k ---"
  python3 "$REPO/longctx_probe.py" 8117 "$tag-48k" 48000 32 "$spec" 2>&1 | sed 's/^/  /'
}

run_arm () {  # tag spec extra_env...
  local tag=$1 spec=$2; shift 2
  echo; echo "############ ARM $tag  SPEC=$spec  $*  $(date -u +%H:%M:%S) ############"
  stop_srv
  env -u SPEC -u VLLM_ROCM_USE_AITER -u VLLM_DISABLE_COMPILE_CACHE -u AITER_CONFIG_GEMM_A8W8 \
      -u VLLM_EXTRA_ARGS -u VLLM_ROCM_SPLITKV_PA \
      PORT=8117 SPEC="$spec" "$@" bash "$LAUNCH" || { echo "[$tag] ❌ LAUNCH FAILED"; return 1; }
  local t0; t0=$(date +%s)
  for i in $(seq 1 240); do curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 && break; sleep 5; done
  if ! curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1; then
    local S; S=$(ls -t /home/qiba/ai/logs/ornith397b/server-8117-*.log | head -1)
    echo "[$tag] ❌ NOT READY："; grep -aoE "(ValueError|RuntimeError|AssertionError|ImportError): .{0,160}" "$S" | sort -u | head -4
    return 1
  fi
  echo "  ✅ ready $(( $(date +%s)-t0 ))s"
  local S; S=$(ls -t /home/qiba/ai/logs/ornith397b/server-8117-*.log | head -1)
  echo "  日志 $S"
  echo "  生效证据：Selected=$(grep -ac 'Selected AiterInt8ScaledMMLinearKernel' "$S")" \
       "TritonFallback=$(grep -ac 'Selected TritonInt8ScaledMMLinearKernel' "$S")" \
       "attn后端=$(grep -aoE 'Overriding with [A-Za-z_]+|Using [A-Za-z_]*ATTN[A-Za-z_]*' "$S" | sort -u | tr '\n' ',')" \
       "notfound=$(grep -ac 'not found tuned config' "$S")" \
       "KV=$(grep -aoE 'GPU KV cache size: [0-9,]+' "$S" | head -1)"
  measure "$tag" "$spec"
}

run_arm attn-triton  5 VLLM_EXTRA_ARGS="--attention-backend TRITON_ATTN"
run_arm spec0-splitkv 0 VLLM_ROCM_SPLITKV_PA=1
run_arm attn-rocm    5 VLLM_EXTRA_ARGS="--attention-backend ROCM_ATTN"
echo; echo "=== 完成：最后留运行的是 attn-rocm（显式 ROCM_ATTN = 默认 auto 的落点，与出厂配方等价）$(date -u +%FT%TZ) ==="
