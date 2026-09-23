#!/usr/bin/env bash
# A 方案主实验：align vs all（+ 各自的数值硬门 + 长回答多轮画像）。
# 顺序：① align 长回答画像 + V2 硬门  ② apply port_mamba_all.py  ③ all 同上  ④ 恢复 8117 出厂态
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int8w8a8attn_aiter_vllm_rocm72_mtp5_256k_8117_ornith_mi250dx8.sh
source "$REPO/_guard.sh"
exec > >(tee -a "$REPO/logs/mamba_all_ab.log") 2>&1
echo "=== mamba all A/B start $(date -u +%FT%TZ)  MY_PORT=$MY_PORT ==="

stop_8117_mine () {
  for p in $(pgrep -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true); do
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || true)
    if echo "$cmd" | grep -q -- "--port 8117" && echo "$cmd" | grep -q "Ornith-1.5-397B-Quark-Int8-Attn"; then
      echo "停本会话 8117 服务 pid=$p"; kill -TERM -"$p" 2>/dev/null
    fi
  done
  sleep 20
}

measure_all () {  # tag
  local tag=$1
  echo "  --- V2 数值硬门（命中复用 vs 冷算：输出 token + logprobs）---"
  python3 "$REPO/reuse_consistency.py" 8127 "$tag" 64 1e-2 2>&1 | sed 's/^/    /'
  echo "  --- 长回答多轮（4 轮 × 1500 tok ≈ 2.8 块/轮）---"
  python3 "$REPO/multiturn_probe.py" 8127 "$tag" 4 1500 2>&1 | sed 's/^/    /'
  echo "  --- 短回答多轮（4 轮 × 64 tok）---"
  python3 "$REPO/multiturn_probe.py" 8127 "$tag-short" 4 64 2>&1 | sed 's/^/    /'
  echo "  --- 单流 count（n=3）---"
  python3 "$REPO/measure_median.py" 8127 "$tag" 3 256 count 2>&1 | tail -1 | sed 's/^/    /'
  echo "  --- 长 prompt 冷/暖 TTFT ---"
  python3 "$REPO/prefix_probe.py" 8127 "$tag-pfx" 2>&1 | grep -aE "冷请求|暖请求|hits|queries|命中率" | sed 's/^/    /'
}

run_arm () {  # tag spec extra env_extra
  local tag=$1 spec=$2 extra=$3 envx=${4:-}
  echo; echo "######## ARM $tag  SPEC=$spec EXTRA='$extra' ENV='$envx'  $(date -u +%H:%M:%S) ########"
  stop_mine; wait_released
  guard_no_other_servers || { echo "⛔ 有别的会话在服务，退出"; return 1; }
  guard_vram_free || return 1
  export SPEC="$spec" VLLM_EXTRA_ARGS="$extra" MY_PORT=8127 PORT=8127
  export PID_FILE="/home/qiba/ai/logs/ornith397b-8127-dsh.pid"
  export LOG_FILE="/home/qiba/ai/logs/ornith397b/server-8127-dsh-$(date +%Y%m%d-%H%M).log"
  if [ -n "$envx" ]; then export VLLM_QWEN35_MAMBA_ALL="$envx"; else unset VLLM_QWEN35_MAMBA_ALL; fi
  bash "$LAUNCH" || { echo "[$tag] ❌ LAUNCH FAILED"; return 1; }
  local t0=$(( $(date +%s) ))
  for i in $(seq 1 240); do curl -sf http://127.0.0.1:8127/health >/dev/null 2>&1 && break; sleep 5; done
  if ! curl -sf http://127.0.0.1:8127/health >/dev/null 2>&1; then
    echo "[$tag] ❌ NOT READY；异常："
    grep -aoE "(ValueError|RuntimeError|NotImplementedError|AssertionError): .{0,160}" "$LOG_FILE" | sort -u | head -4 | sed 's/^/    /'
    return 1
  fi
  echo "  ✅ ready $(( $(date +%s)-t0 ))s  日志 $LOG_FILE"
  echo "  --- 配置生效证据（'all' 臂必须出现 'all'）---"
  grep -aoE "enable_prefix_caching=[A-Za-z]+|Mamba cache mode is set to '[a-z]+'|[a-z_]*cache mode is set to '[a-z]+'|GPU KV cache size: [0-9,]+ tokens" "$LOG_FILE" | sort -u | sed 's/^/    /'
  measure_all "$tag"
}

stop_8117_mine
# ① align
run_arm align-long 5 "--enable-prefix-caching" ""
# ② 打补丁（三处）
echo; echo "######## apply port_mamba_all.py ########"
python3 /home/qiba/ai/recipes/patches/gfx90a/port_mamba_all.py --env vllm_0.28.0_rocm72 --apply 2>&1 | tail -12 | sed 's/^/  /'
echo "######## post-apply --check ########"
python3 /home/qiba/ai/recipes/patches/gfx90a/port_mamba_all.py --env vllm_0.28.0_rocm72 --check 2>&1 | tail -8 | sed 's/^/  /'
# ③ all
run_arm all-long 5 "--enable-prefix-caching --mamba-cache-mode all" "1"

# ④ 收尾：恢复出厂态到 8117
echo; echo "######## 收尾：恢复出厂态到 8117 ########"
stop_mine; wait_released
if pgrep -af "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
  echo "⛔ 还有 vLLM 进程（可能是别的会话）⇒ 不抢，收尾跳过"; exit 0
fi
unset VLLM_EXTRA_ARGS VLLM_QWEN35_MAMBA_ALL
PORT=8117 SPEC=5 bash "$LAUNCH" || echo "⚠️ 8117 起服失败"
for i in $(seq 1 240); do curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 && break; sleep 5; done
curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 && echo "✅ 8117 出厂态已恢复"
echo "=== done $(date -u +%FT%TZ) ==="
