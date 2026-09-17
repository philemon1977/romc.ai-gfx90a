#!/usr/bin/env bash
# 三臂：前缀缓存 ON×SPEC5 / ON×SPEC0 / 出厂态(OFF×SPEC5)。
# 目的：① 老结论"MTP 一开 hits=0"现在是否仍成立 ② SPEC=0 能否恢复 ③ 恢复出厂态收尾
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int8w8a8attn_aiter_vllm_rocm72_mtp5_256k_8117_ornith_mi250dx8.sh
PIDF=/home/qiba/ai/logs/ornith397b-8117.pid
exec > >(tee -a "$REPO/logs/prefix_mtp_ab.log") 2>&1
echo "=== 前缀缓存 × MTP 三臂 $(date -u +%FT%TZ) ==="

stop_srv () {
  if [ -f "$PIDF" ]; then
    local p; p=$(cat "$PIDF" 2>/dev/null || true)
    [ -n "${p:-}" ] && kill -0 "$p" 2>/dev/null && { kill -TERM -"$p" 2>/dev/null; echo "  已 TERM $p"; }
  fi
  rm -f "$PIDF"
}
wait_free () {
  for _ in $(seq 1 120); do
    nc -z 127.0.0.1 8117 2>/dev/null && { sleep 5; continue; }
    free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    [ "${free:-0}" -ge 62 ] && return 0
    sleep 5
  done
  return 1
}

run_arm () {  # tag spec extra
  local tag=$1 spec=$2 extra=$3
  echo; echo "######## ARM $tag  SPEC=$spec  VLLM_EXTRA_ARGS='$extra'  $(date -u +%H:%M:%S) ########"
  stop_srv; wait_free || echo "  ⚠️ 显存回收慢，仍尝试起服"

  export PORT=8117 SPEC="$spec" VLLM_EXTRA_ARGS="$extra"
  bash "$LAUNCH" || { echo "[$tag] ❌ LAUNCH FAILED"; return 1; }
  local t0=$(( $(date +%s) ))
  for i in $(seq 1 240); do curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 && break; sleep 5; done
  curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 || { echo "[$tag] ❌ NOT READY"; return 1; }
  echo "  ✅ ready $(( $(date +%s)-t0 ))s"
  local S; S=$(ls -t /home/qiba/ai/logs/ornith397b/server-8117-*.log | head -1)
  echo "  --- 配置生效证据 ---"
  grep -aoE "Mamba cache mode is set to '[a-z]+'[^\"]{0,40}|enable_prefix_caching=[A-Za-z]+|mamba_cache_mode='?[a-z]+|Mamba cache mode is set to 'none'[^\"]{0,40}|GPU KV cache size: [0-9,]+ tokens" "$S" | sort -u | sed 's/^/    /'
  echo "  --- 前缀缓存探针 ---"
  python3 "$REPO/prefix_probe.py" 8117 "$tag" 2>&1 | sed 's/^/    /'
  echo "  --- 单流 TPS: count n=3 ---"
  python3 "$REPO/measure_median.py" 8117 "$tag" 3 256 count 2>&1 | tail -2 | sed 's/^/    /'
  echo "  --- 步时分解 count ---"
  python3 "$REPO/step_probe.py" 8117 "$tag" "$spec" count 256 2>&1 | tail -1 | sed 's/^/    /'
}

run_arm pfx-ON-spec5   5 "--enable-prefix-caching"
run_arm pfx-ON-spec0   0 "--enable-prefix-caching"
run_arm shipped-OFF-spec5 5 ""      # 恢复出厂态（不带任何 extra）

echo; echo "=== 完成：已恢复出厂态（前缀缓存关 + SPEC=5），服务保持运行 $(date -u +%FT%TZ) ==="
