#!/usr/bin/env bash
# 验证"问题2"的修复：qwen4_exp 草稿组标注 ⇒ 恢复跨请求前缀复用。
# 四步：停服（只按 pid）→ 带 VLLM_QWEN4EXP_EAGLE_ANNOTATE=1 起服 → 自证（警告消失+patch 行出现）
#       → 数值硬门 + 多轮画像 + 单流回归。末尾**保持服务运行**。
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/mnt/stripe-3mix-3t2/models/Qwen/launcher/qwen3.8-flash-next_176b_bf16_vllm_rocm724_256k_8107_Qwen_mi250dx8.sh
PIDF=/home/qiba/ai/logs/qwen3.8-flash-next-8107.pid
exec > >(tee -a "$REPO/logs/qwen38_fix2.log") 2>&1
echo "=== 问题2 验证（草稿组标注）  $(date -u +%FT%TZ) ==="

stop_mine () {
  local p
  [ -f "$PIDF" ] || { echo "  （无 PID 文件）"; return 0; }
  p=$(cat "$PIDF" 2>/dev/null || true); rm -f "$PIDF"
  [ -n "${p:-}" ] || return 0
  if kill -0 "$p" 2>/dev/null && ! tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- "--port 8107"; then
    echo "  ⛔ pid $p 不是 8107 服务，拒绝杀"; return 1
  fi
  # 只按**具体 pid** 杀（不用负号进程组：本会话实测它会连带打死调用方自己的 shell）
  for x in $p $(rocm-smi --showpids 2>/dev/null | awk '/VLLM::Worker|EngineCor/{print $1}'); do
    kill -TERM "$x" 2>/dev/null
  done
  local i; for i in $(seq 1 5); do kill -0 "$p" 2>/dev/null || break; sleep 5; done
  if kill -0 "$p" 2>/dev/null; then
    echo "  TERM 25s 未生效 ⇒ 逐个 KILL（本底座 TERM 会挂）"
    for x in $p $(rocm-smi --showpids 2>/dev/null | awk '/VLLM::Worker|EngineCor/{print $1}'); do
      kill -KILL "$x" 2>/dev/null
    done
    sleep 8
  fi
  return 0
}

stop_mine
for i in $(seq 1 60); do
  wk=$(rocm-smi --showpids 2>/dev/null | grep -c "VLLM::Worker" || true)
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "${free:-0}" -ge 58 ] && [ "${wk:-1}" = "0" ] && { echo "  已释放（min free ${free} GiB, worker ${wk}, $((i*5))s）"; break; }
  sleep 5
done

echo "=== 起服（VLLM_QWEN4EXP_EAGLE_ANNOTATE=1）==="
export VLLM_QWEN4EXP_EAGLE_ANNOTATE=1
PORT=8107 bash "$LAUNCH" || { echo "❌ LAUNCH FAILED"; exit 1; }
for i in $(seq 1 200); do curl -sf http://127.0.0.1:8107/health >/dev/null 2>&1 && break; sleep 5; done
curl -sf http://127.0.0.1:8107/health >/dev/null 2>&1 || { echo "❌ NOT READY"; exit 1; }
echo "  ✅ ready"
L=$(cat /home/qiba/ai/logs/qwen3.8-flash-next-8107.logpath 2>/dev/null)   # 2026-09-18：日志改落 logs/flash-next/，认侧车而不是 glob（旧 glob 会静默读到旧日志）
[ -n "$L" ] && [ -r "$L" ] || L=$(ls -t /home/qiba/ai/logs/qwen3.8-flash-next_*8107*.log /home/qiba/ai/logs/flash-next/server-8107-*.log 2>/dev/null | head -1)
echo "  --- 自证 ---"
echo "   patch 生效行: $(grep -ac 'port_qwen4exp_eagle_annotate' "$L") 次"
echo "   ⛔ 兜底警告行(应为 0): $(grep -ac 'will be treated as a draft group' "$L") 次"
grep -aoE "\[port_qwen4exp_eagle_annotate\][^\"]{0,80}" "$L" | head -1 | sed 's/^/   /'
grep -aoE "GPU KV cache size: [0-9,]+ tokens|Mamba cache mode is set to '[a-z]+'" "$L" | sort -u | sed 's/^/   /'

echo; echo "=== 数值硬门（命中复用 vs 冷算）==="
python3 "$REPO/qwen38_agent_probe.py" 8107 qwen3.8-flash-next consistency 2>&1 | sed 's/^/  /'
echo; echo "=== 多轮画像（agent 式，上下文累积）==="
python3 "$REPO/qwen38_agent_probe.py" 8107 qwen3.8-flash-next multiturn 4 400 2>&1 | sed 's/^/  /'
echo; echo "=== 单流回归（n=5）==="
python3 "$REPO/qwen38_agent_probe.py" 8107 qwen3.8-flash-next singlestream 2>&1 | sed 's/^/  /'
echo "=== done，服务保持运行 $(date -u +%FT%TZ) ==="
