#!/usr/bin/env bash
# 最终判定：带补丁 env（复用开启）跑 consistency3（真冷基准），然后**恢复出厂态**（不带 env）。
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/mnt/stripe-3mix-3t2/models/Qwen/launcher/qwen3.8-flash-next_176b_bf16_vllm_rocm724_256k_8107_Qwen_mi250dx8.sh
PIDF=/home/qiba/ai/logs/qwen3.8-flash-next-8107.pid
exec > >(tee -a "$REPO/logs/qwen38_fix2_final.log") 2>&1
echo "=== 最终判定 + 恢复出厂态  $(date -u +%FT%TZ) ==="

stop_mine () {
  local p; p=$(cat "$PIDF" 2>/dev/null || true); rm -f "$PIDF"
  [ -n "${p:-}" ] && kill -0 "$p" 2>/dev/null || { echo "  （无本会话服务）"; return 0; }
  for x in $p $(rocm-smi --showpids 2>/dev/null | awk '/VLLM::Worker|EngineCor/{print $1}'); do kill -TERM "$x" 2>/dev/null; done
  local i; for i in $(seq 1 5); do kill -0 "$p" 2>/dev/null || break; sleep 5; done
  if kill -0 "$p" 2>/dev/null; then
    for x in $p $(rocm-smi --showpids 2>/dev/null | awk '/VLLM::Worker|EngineCor/{print $1}'); do kill -KILL "$x" 2>/dev/null; done
    sleep 8
  fi
}
wait_free () {
  local i wk free
  for i in $(seq 1 60); do
    wk=$(rocm-smi --showpids 2>/dev/null | grep -c "VLLM::Worker" || true)
    free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    [ "${free:-0}" -ge 58 ] && [ "${wk:-1}" = "0" ] && { echo "  已释放（$free GiB, worker $wk）"; return 0; }
    sleep 5
  done
}
ready () { local i; for i in $(seq 1 200); do curl -sf http://127.0.0.1:8107/health >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }

echo; echo "######## ① 带补丁 env：真冷基准 ########"
stop_mine; wait_free
export VLLM_QWEN4EXP_EAGLE_ANNOTATE=1
unset VLLM_EXTRA_ARGS
PORT=8107 bash "$LAUNCH" || { echo "❌ LAUNCH FAILED"; exit 1; }
ready || { echo "❌ NOT READY"; exit 1; }
echo "  ✅ ready"
L=$(cat /home/qiba/ai/logs/qwen3.8-flash-next-8107.logpath 2>/dev/null)   # 2026-09-18：日志改落 logs/flash-next/，认侧车而不是 glob（旧 glob 会静默读到旧日志）
[ -n "$L" ] && [ -r "$L" ] || L=$(ls -t /home/qiba/ai/logs/qwen3.8-flash-next_*8107*.log /home/qiba/ai/logs/flash-next/server-8107-*.log 2>/dev/null | head -1)
echo "  兜底警告(应 0): $(grep -ac 'will be treated as a draft group' "$L")   patch 行: $(grep -ac port_qwen4exp_eagle_annotate "$L")"
python3 "$REPO/qwen38_agent_probe.py" 8107 qwen3.8-flash-next consistency3 2>&1 | sed 's/^/  /'

echo; echo "######## ② 多轮命中画像（复用开启）########"
python3 "$REPO/qwen38_agent_probe.py" 8107 qwen3.8-flash-next multiturn 4 400 2>&1 | sed 's/^/  /'

echo; echo "######## ③ 恢复出厂态（不带 env；前缀缓存按出厂默认开）########"
stop_mine; wait_free
unset VLLM_QWEN4EXP_EAGLE_ANNOTATE VLLM_EXTRA_ARGS
PORT=8107 bash "$LAUNCH" || { echo "⚠️ 恢复起服失败"; exit 1; }
ready && echo "  ✅ 出厂态已恢复（补丁仍在源码里，但未设 env ⇒ 与上游行为一致）" || echo "  ⚠️ 未就绪"
echo "=== done $(date -u +%FT%TZ) ==="
