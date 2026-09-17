#!/usr/bin/env bash
# 多轮前缀复用画像：align（前缀缓存开）vs off（出厂态），都跑在**本会话专用端口 8127**。
# 全程 source _guard.sh：起服前硬门（无其它 api_server、8 die 全空）、停服只杀自己记录过的 pid。
# 末尾**恢复出厂态到 8117**（用户的服务端口），并再次确认没有踩到别的会话。
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int8w8a8attn_aiter_vllm_rocm72_mtp5_256k_8117_ornith_mi250dx8.sh
source "$REPO/_guard.sh"
exec > >(tee -a "$REPO/logs/multiturn_ab.log") 2>&1
echo "=== 多轮复用画像 start $(date -u +%FT%TZ)  MY_PORT=$MY_PORT PID_FILE=$PID_FILE ==="

# 先停本会话可能已在跑的服务（8117 是我的 arm3；核对端口与模型后再停）
for p in $(pgrep -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || true)
  if echo "$cmd" | grep -q -- "--port 8117" && echo "$cmd" | grep -q "Ornith-1.5-397B-Quark-Int8-Attn"; then
    echo "停本会话 8117 服务 pid=$p"
    kill -TERM -"$p" 2>/dev/null
  fi
done
sleep 20
guard_all || { echo "⛔ guard 拦住：不抢卡，退出"; exit 1; }

run_arm () {  # tag spec extra
  local tag=$1 spec=$2 extra=$3
  echo; echo "######## ARM $tag  SPEC=$spec  EXTRA='$extra'  $(date -u +%H:%M:%S) ########"
  stop_mine; wait_released
  guard_no_other_servers || { echo "⛔ 有别的会话在服务，退出"; return 1; }
  guard_vram_free || return 1

  export SPEC="$spec" VLLM_EXTRA_ARGS="$extra" MY_PORT=8127 PORT=8127
  export PID_FILE="/home/qiba/ai/logs/ornith397b-8127-dsh.pid"
  export LOG_FILE="/home/qiba/ai/logs/ornith397b/server-8127-dsh-$(date +%Y%m%d-%H%M).log"
  bash "$LAUNCH" || { echo "[$tag] ❌ LAUNCH FAILED"; return 1; }
  local t0=$(( $(date +%s) ))
  for i in $(seq 1 240); do curl -sf http://127.0.0.1:8127/health >/dev/null 2>&1 && break; sleep 5; done
  curl -sf http://127.0.0.1:8127/health >/dev/null 2>&1 || { echo "[$tag] ❌ NOT READY"; return 1; }
  echo "  ✅ ready $(( $(date +%s)-t0 ))s  日志 $LOG_FILE"
  grep -aoE "enable_prefix_caching=[A-Za-z]+|Mamba cache mode is set to '[a-z]+'|GPU KV cache size: [0-9,]+ tokens" "$LOG_FILE" | sort -u | sed 's/^/    /'
  echo "  --- 多轮画像（5 轮，每轮生成 160 tok）---"
  python3 "$REPO/multiturn_probe.py" 8127 "$tag" 5 160 2>&1 | sed 's/^/    /'
  echo "  --- 单流 TPS ---"
  python3 "$REPO/measure_median.py" 8127 "$tag" 3 256 count 2>&1 | tail -1 | sed 's/^/    /'
  echo "  --- 长 prompt 单次 TTFT（≈2.8k prompt）---"
  python3 "$REPO/prefix_probe.py" 8127 "$tag-pfx" 2>&1 | grep -aE "冷请求|暖请求|hits|queries|命中率" | sed 's/^/    /'
}

run_arm align-ON  5 "--enable-prefix-caching"
run_arm off-ship  5 ""

# ── 收尾：恢复出厂态到 8117（用户的服务端口），并核对没有踩到别人 ──
echo; echo "######## 收尾：恢复出厂态到 8117 ########"
stop_mine; wait_released
if pgrep -af "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
  echo "⛔ 还有 vLLM 进程在跑（可能是别的会话）⇒ 按规矩不抢，收尾跳过；请稍后手动起 8117"
  exit 0
fi
unset VLLM_EXTRA_ARGS
PORT=8117 SPEC=5 bash "$LAUNCH" || echo "⚠️ 8117 起服失败"
for i in $(seq 1 240); do curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 && break; sleep 5; done
curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 && echo "✅ 8117 出厂态已恢复（前缀缓存关 + SPEC=5）"
echo "=== done $(date -u +%FT%TZ) ==="
