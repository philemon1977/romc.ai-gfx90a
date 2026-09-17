#!/usr/bin/env bash
# 决定性实验：真实 vibe-coding 形状下的多轮复用（每轮回答 ~1500 tok，即 ~2.8 个 block）。
# 判定标准（'all' 是否必需）：
#   若 align 下命中仍 ≈ floor(与上轮共享长度/544) − 1 块（即把**上轮生成的回答**也复用上了）
#     ⇒ 'all' 不必要，收益已经拿到，只需在 agent 档打开前缀缓存（零风险）。
#   若命中停在"上轮 prefill 的末尾"（每轮丢掉 ≈ 上轮回答的长度）
#     ⇒ 'all' 必需，按 docs/Ornith-397B-Mamba-All-模式实现方案 走 M1→M4。
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int8w8a8attn_aiter_vllm_rocm72_mtp5_256k_8117_ornith_mi250dx8.sh
source "$REPO/_guard.sh"
exec > >(tee -a "$REPO/logs/multiturn_long.log") 2>&1
echo "=== 长回答多轮实验 start $(date -u +%FT%TZ)  MY_PORT=$MY_PORT ==="

stop_sessions_8117 () {
  for p in $(pgrep -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true); do
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || true)
    if echo "$cmd" | grep -q -- "--port 8117" && echo "$cmd" | grep -q "Ornith-1.5-397B-Quark-Int8-Attn"; then
      echo "停本会话 8117 服务 pid=$p"; kill -TERM -"$p" 2>/dev/null
    fi
  done
  sleep 20
}
stop_sessions_8117
stop_mine; wait_released

run_arm () {  # tag spec extra turns ans
  local tag=$1 spec=$2 extra=$3 turns=$4 ans=$5
  echo; echo "######## ARM $tag  SPEC=$spec EXTRA='$extra' turns=$turns ans=$ans  $(date -u +%H:%M:%S) ########"
  guard_no_other_servers || { echo "⛔ 有别的会话在服务，退出"; return 1; }
  guard_vram_free || return 1
  export SPEC="$spec" VLLM_EXTRA_ARGS="$extra" MY_PORT=8127 PORT=8127
  export PID_FILE="/home/qiba/ai/logs/ornith397b-8127-dsh.pid"
  export LOG_FILE="/home/qiba/ai/logs/ornith397b/server-8127-dsh-$(date +%Y%m%d-%H%M).log"
  bash "$LAUNCH" || { echo "[$tag] ❌ LAUNCH FAILED"; return 1; }
  local t0=$(( $(date +%s) ))
  for i in $(seq 1 240); do curl -sf http://127.0.0.1:8127/health >/dev/null 2>&1 && break; sleep 5; done
  curl -sf http://127.0.0.1:8127/health >/dev/null 2>&1 || { echo "[$tag] ❌ NOT READY"; return 1; }
  echo "  ✅ ready $(( $(date +%s)-t0 ))s"
  grep -aoE "enable_prefix_caching=[A-Za-z]+|Mamba cache mode is set to '[a-z]+'|GPU KV cache size: [0-9,]+ tokens" "$LOG_FILE" | sort -u | sed 's/^/    /'
  echo "  --- 长回答多轮（$turns 轮 × $ans tok 回答）---"
  python3 "$REPO/multiturn_probe.py" 8127 "$tag" "$turns" "$ans" 2>&1 | sed 's/^/    /'
  echo "  --- 对照：同臂短回答（4 轮 × 64 tok）---"
  python3 "$REPO/multiturn_probe.py" 8127 "$tag-short" 4 64 2>&1 | sed 's/^/    /'
}

run_arm align-long 5 "--enable-prefix-caching" 4 1500

# 收尾：恢复出厂态到 8117
echo; echo "######## 收尾：恢复出厂态到 8117 ########"
stop_mine; wait_released
if pgrep -af "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
  echo "⛔ 还有 vLLM 进程（可能是别的会话）⇒ 按规矩不抢，收尾跳过"; exit 0
fi
unset VLLM_EXTRA_ARGS
PORT=8117 SPEC=5 bash "$LAUNCH" || echo "⚠️ 8117 起服失败"
for i in $(seq 1 240); do curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 && break; sleep 5; done
curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 && echo "✅ 8117 出厂态已恢复"
echo "=== done $(date -u +%FT%TZ) ==="
