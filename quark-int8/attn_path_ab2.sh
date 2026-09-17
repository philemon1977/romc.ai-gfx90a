#!/usr/bin/env bash
# 接管在飞的 attn-triton 臂（前一个 runner 被工具超时的进程组信号带走），
# 然后继续跑 spec0-splitkv 与 attn-rocm。**本脚本自身用 setsid 启动**，不受父进程组信号影响。
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int8w8a8attn_aiter_vllm_rocm72_mtp5_256k_8117_ornith_mi250dx8.sh
PIDF=/home/qiba/ai/logs/ornith397b-8117.pid
exec > >(tee -a "$REPO/logs/attn_path_ab.log") 2>&1
echo "=== runner 重新接管 $(date -u +%FT%TZ)（前一个 runner 被父进程组信号带走，服务本身存活）==="

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

evidence_and_measure () {  # tag spec
  local tag=$1 spec=$2
  local S; S=$(ls -t /home/qiba/ai/logs/ornith397b/server-8117-*.log | head -1)
  echo "  日志 $S"
  echo "  生效证据：AiterInt8=$(grep -ac 'Selected AiterInt8ScaledMMLinearKernel' "$S")" \
       "TritonInt8=$(grep -ac 'Selected TritonInt8ScaledMMLinearKernel' "$S")" \
       "attn=$(grep -aoE 'Overriding with [A-Za-z_]+|Using [A-Za-z_]*ATTN[A-Za-z_]*' "$S" | sort -u | tr '\n' ',')" \
       "notfound=$(grep -ac 'not found tuned config' "$S")" \
       "KV=$(grep -aoE 'GPU KV cache size: [0-9,]+' "$S" | head -1 | grep -oE '[0-9,]+')"
  echo "--- [$tag] 短上下文 count n=5 ---"
  python3 "$REPO/measure_median.py" 8117 "$tag-count" 5 256 count 2>&1 | sed 's/^/  /'
  echo "--- [$tag] 短上下文 explain n=5 ---"
  python3 "$REPO/measure_median.py" 8117 "$tag-explain" 5 256 explain 2>&1 | sed 's/^/  /'
  echo "--- [$tag] 步时分解（短上下文 count, SPEC=$spec）---"
  python3 "$REPO/step_probe.py" 8117 "$tag" "$spec" count 256 2>&1 | sed 's/^/  /'
  echo "--- [$tag] 长上下文 @16.7k ---"
  python3 "$REPO/longctx_probe.py" 8117 "$tag-16k" 16000 32 "$spec" 2>&1 | sed 's/^/  /'
  echo "--- [$tag] 长上下文 @48k ---"
  python3 "$REPO/longctx_probe.py" 8117 "$tag-48k" 48000 32 "$spec" 2>&1 | sed 's/^/  /'
}

wait_ready () {  # tag
  local t0; t0=$(date +%s)
  for i in $(seq 1 240); do curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 && break; sleep 5; done
  if ! curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1; then
    local S; S=$(ls -t /home/qiba/ai/logs/ornith397b/server-8117-*.log | head -1)
    echo "[$1] ❌ NOT READY："; grep -aoE "(ValueError|RuntimeError|AssertionError|ImportError): .{0,160}" "$S" | sort -u | head -4
    return 1
  fi
  echo "  ✅ ready $(( $(date +%s)-t0 ))s"
}

run_arm () {  # tag spec extra_env...
  local tag=$1 spec=$2; shift 2
  echo; echo "############ ARM $tag  SPEC=$spec  $*  $(date -u +%H:%M:%S) ############"
  stop_srv
  env -u SPEC -u VLLM_ROCM_USE_AITER -u VLLM_DISABLE_COMPILE_CACHE -u AITER_CONFIG_GEMM_A8W8 \
      -u VLLM_EXTRA_ARGS -u VLLM_ROCM_SPLITKV_PA \
      PORT=8117 SPEC="$spec" "$@" bash "$LAUNCH" || { echo "[$tag] ❌ LAUNCH FAILED"; return 1; }
  wait_ready "$tag" || return 1
  evidence_and_measure "$tag" "$spec"
}

# ① 接管在飞的 attn-triton（只等就绪，不重启）
echo; echo "############ ARM attn-triton (接管，SPEC=5, --attention-backend TRITON_ATTN) $(date -u +%H:%M:%S) ############"
if wait_ready attn-triton; then evidence_and_measure attn-triton 5; else echo "接管失败，跳过"; fi

# ② SPEC=0 + splitKV（长上下文真杠杆）
run_arm spec0-splitkv 0 VLLM_ROCM_SPLITKV_PA=1

# ③ 显式 ROCM_ATTN（=默认 auto 的落点；收尾留运行）
run_arm attn-rocm 5 VLLM_EXTRA_ARGS="--attention-backend ROCM_ATTN"

echo; echo "=== 完成：留运行 attn-rocm（显式 ROCM_ATTN，与出厂默认等价）$(date -u +%FT%TZ) ==="
