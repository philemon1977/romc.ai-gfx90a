#!/usr/bin/env bash
# INT8-Attn 提速臂对照：把"本会话三个有效杠杆"逐个装上并在**同口径**下量。
#   A int8aiter-mtp5 : AITER=1 SPEC=5 no-pfx   ← 8117 出厂配方（主交付）
#   B int8triton-mtp5: AITER=0 SPEC=5 no-pfx   ← 单变量对照（证明 +?% 来自 aiter）
#   C int8aiter-mtp3 : AITER=1 SPEC=3 no-pfx   ← SPEC=5 质量测不了，SPEC=3 出代理质量(NLL)
# 口径：固定 prompt + 丢弃首个请求 + 中位数(n=5)；count 与 explain 两个 workload 都报。
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int8w8a8attn_aiter_vllm_rocm72_mtp5_256k_8117_ornith_mi250dx8.sh
PIDF=/home/qiba/ai/logs/ornith397b-8117.pid
LOG="$REPO/logs/int8aiter_ab.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== int8aiter_ab start $(date -u +%FT%TZ) ==="

stop_srv () {
  if [ -f "$PIDF" ]; then
    local p; p=$(cat "$PIDF" 2>/dev/null || true)
    [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null && echo "   已 TERM 上一实例 $p"
  fi
  rm -f "$PIDF"
}

wait_free () {
  for _ in $(seq 1 120); do
    nc -z 127.0.0.1 8117 2>/dev/null && { sleep 10; continue; }
    free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    [ "${free:-0}" -ge 62 ] && { echo "   显存已回收（min free ${free} GiB）"; return 0; }
    sleep 10
  done
  echo "   ⚠️ 显存一直不空（min free ${free:-?} GiB）"; return 1
}

run_arm () {  # tag spec aiter do_nll
  local tag=$1 spec=$2 aiter=$3 do_nll=${4:-0}
  echo; echo "############ ARM $tag  SPEC=$spec AITER=$aiter  $(date -u +%H:%M:%S) ############"
  stop_srv; wait_free || true

  export PORT=8117 SPEC="$spec" VLLM_ROCM_USE_AITER="$aiter"
  unset VLLM_EXTRA_ARGS
  bash "$LAUNCH" || { echo "[$tag] ❌ LAUNCH FAILED"; return 1; }

  local t0 i ready=0
  t0=$(date +%s)
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 && { ready=1; break; }
    sleep 5
  done
  local srv; srv=$(ls -t /home/qiba/ai/logs/ornith397b/server-8117-*.log | head -1)
  echo "   服务日志：$srv"
  if [ "$ready" != "1" ]; then
    echo "[$tag] ❌ NOT READY（$(( $(date +%s)-t0 ))s）："
    grep -aoE "(ValueError|RuntimeError|AssertionError|AttributeError|ImportError|Consumer error): .{0,170}" "$srv" | sort -u | head -5
    return 1
  fi
  echo "   ✅ ready，用时 $(( $(date +%s)-t0 ))s"
  echo "--- 生效证据（缺 AiterInt8ScaledMMLinearKernel 即没吃上 aiter）---"
  grep -aoE "Selected [A-Za-z0-9]*Int8ScaledMMLinearKernel|\[aiter\] import \[module_gemm_a8w8\]|\[gfx90a-patch\][^\"]{0,60}|Using [A-Za-z_']+ Int8 MoE backend|GPU KV cache size: [0-9,]+ tokens|Loading weights took [0-9.]+ seconds|Application startup complete" "$srv" | sort | uniq -c | sort -rn | sed 's/^/  /'

  echo "--- 单流 TPS：count (n=5) ---"
  python3 "$REPO/measure_median.py" 8117 "$tag" 5 256 count 2>&1 | sed 's/^/  /'
  echo "--- 单流 TPS：explain (n=5) ---"
  python3 "$REPO/measure_median.py" 8117 "$tag" 5 256 explain 2>&1 | sed 's/^/  /'
  echo "--- 投机解码计数 ---"
  curl -s localhost:8117/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{|spec_decode_(draft|accepted)_tokens" | sed 's/^/  /' | head -6
  if [ "$do_nll" = "1" ]; then
    echo "--- 质量（SPEC=$spec，探针自带 nll>8 拒收门）---"
    python3 "$REPO/nll_probe.py" 8117 ornith "$tag" 2>&1 | sed 's/^/  /'
    echo "  (复测一次以确认可复现)"
    python3 "$REPO/nll_probe.py" 8117 ornith "$tag-rep2" 2>&1 | sed 's/^/  /'
  fi
  stop_srv; echo "[$tag] 已停服"; sleep 25
}

run_arm int8aiter-mtp5  5 1 0
run_arm int8triton-mtp5 5 0 0
run_arm int8aiter-mtp3  3 1 1
echo; echo "=== int8aiter_ab done $(date -u +%FT%TZ) ==="
