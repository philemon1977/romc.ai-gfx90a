#!/usr/bin/env bash
# 对照臂：关前缀缓存（⇒ 每次请求都全量重算，无任何状态复用）+ **不带** VLLM_QWEN4EXP_EAGLE_ANNOTATE
# ⇒ 用来量"这个引擎在相同输入下的固有非确定性"。若这里三次输出就不同，则"冷 vs 暖必须逐位相同"
# 这条判据对本引擎不成立，硬门必须改成统计口径（见报告）。
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/mnt/stripe-3mix-3t2/models/Qwen/launcher/qwen3.8-flash-next_176b_bf16_vllm_rocm724_256k_8107_Qwen_mi250dx8.sh
PIDF=/home/qiba/ai/logs/qwen3.8-flash-next-8107.pid
exec > >(tee -a "$REPO/logs/qwen38_gate_ctl.log") 2>&1
echo "=== 对照臂：关前缀缓存、无补丁 env  $(date -u +%FT%TZ) ==="

p=$(cat "$PIDF" 2>/dev/null || true)
if [ -n "${p:-}" ] && kill -0 "$p" 2>/dev/null; then
  echo "  停本会话 8107 pid=$p（只按具体 pid）"
  for x in $p $(rocm-smi --showpids 2>/dev/null | awk '/VLLM::Worker|EngineCor/{print $1}'); do kill -TERM "$x" 2>/dev/null; done
  for i in $(seq 1 5); do kill -0 "$p" 2>/dev/null || break; sleep 5; done
  if kill -0 "$p" 2>/dev/null; then
    for x in $p $(rocm-smi --showpids 2>/dev/null | awk '/VLLM::Worker|EngineCor/{print $1}'); do kill -KILL "$x" 2>/dev/null; done
    sleep 8
  fi
fi
rm -f "$PIDF"
for i in $(seq 1 60); do
  wk=$(rocm-smi --showpids 2>/dev/null | grep -c "VLLM::Worker" || true)
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "${free:-0}" -ge 58 ] && [ "${wk:-1}" = "0" ] && { echo "  已释放（min free $free GiB, worker $wk）"; break; }
  sleep 5
done

unset VLLM_QWEN4EXP_EAGLE_ANNOTATE
export VLLM_EXTRA_ARGS="--no-enable-prefix-caching"
PORT=8107 bash "$LAUNCH" || { echo "❌ LAUNCH FAILED"; exit 1; }
for i in $(seq 1 200); do curl -sf http://127.0.0.1:8107/health >/dev/null 2>&1 && break; sleep 5; done
curl -sf http://127.0.0.1:8107/health >/dev/null 2>&1 || { echo "❌ NOT READY"; exit 1; }
echo "  ✅ ready"
L=$(cat /home/qiba/ai/logs/qwen3.8-flash-next-8107.logpath 2>/dev/null)   # 2026-09-18：日志改落 logs/flash-next/，认侧车而不是 glob（旧 glob 会静默读到旧日志）
[ -n "$L" ] && [ -r "$L" ] || L=$(ls -t /home/qiba/ai/logs/qwen3.8-flash-next_*8107*.log /home/qiba/ai/logs/flash-next/server-8107-*.log 2>/dev/null | head -1)
grep -aoE "enable_prefix_caching=[A-Za-z]+" "$L" | sort -u | sed 's/^/   /'
echo; echo "=== 固有非确定性测量：同一 prompt 连发 3 次（全部为冷算，无复用）==="
python3 "$REPO/qwen38_agent_probe.py" 8107 qwen3.8-flash-next consistency2 2>&1 | sed 's/^/  /'
echo "=== done $(date -u +%FT%TZ) ==="
