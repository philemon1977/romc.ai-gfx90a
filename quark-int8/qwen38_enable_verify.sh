#!/usr/bin/env bash
# 端到端验证"已开启"：**不外部设 env**，只用 launcher 默认（护栏⑤ 应自动 export + 断言）
# 验证四项：① 护栏⑤/补丁行出现、兜底警告 0 ② 真冷基准数值硬门 ③ 多轮命中与 TTFT ④ 单流回归
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/mnt/stripe-3mix-3t2/models/Qwen/launcher/qwen3.8-flash-next_176b_bf16_vllm_rocm724_256k_8107_Qwen_mi250dx8.sh
PIDF=/home/qiba/ai/logs/qwen3.8-flash-next-8107.pid
exec > >(tee -a "$REPO/logs/qwen38_enable_verify.log") 2>&1
echo "=== 已开启态端到端验证  $(date -u +%FT%TZ) ==="

p=$(cat "$PIDF" 2>/dev/null || true)
if [ -n "${p:-}" ] && kill -0 "$p" 2>/dev/null; then
  echo "  停本会话 8107 pid=$p"
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
  [ "${free:-0}" -ge 58 ] && [ "${wk:-1}" = "0" ] && { echo "  已释放（$free GiB, worker $wk）"; break; }
  sleep 5
done

echo; echo "=== 用 launcher 默认起服（不外部设 env）==="
unset VLLM_QWEN4EXP_EAGLE_ANNOTATE VLLM_EXTRA_ARGS
PORT=8107 bash "$LAUNCH" || { echo "❌ LAUNCH FAILED"; exit 1; }
for i in $(seq 1 200); do curl -sf http://127.0.0.1:8107/health >/dev/null 2>&1 && break; sleep 5; done
curl -sf http://127.0.0.1:8107/health >/dev/null 2>&1 || { echo "❌ NOT READY"; exit 1; }
echo "  ✅ ready"
L=$(cat /home/qiba/ai/logs/qwen3.8-flash-next-8107.logpath 2>/dev/null)   # 2026-09-18：日志改落 logs/flash-next/，认侧车而不是 glob（旧 glob 会静默读到旧日志）
[ -n "$L" ] && [ -r "$L" ] || L=$(ls -t /home/qiba/ai/logs/qwen3.8-flash-next_*8107*.log /home/qiba/ai/logs/flash-next/server-8107-*.log 2>/dev/null | head -1)
echo "  --- 自证 ---"
echo "   护栏⑤行: $(grep -ac '护栏⑤' "$REPO/logs/qwen38_enable_verify.log")   patch 行: $(grep -ac port_qwen4exp_eagle_annotate "$L")   兜底警告(应0): $(grep -ac 'will be treated as a draft group' "$L")"
grep -aoE "GPU KV cache size: [0-9,]+ tokens" "$L" | tail -1 | sed 's/^/   /'

echo; echo "=== ① 真冷基准数值硬门 ==="
python3 "$REPO/qwen38_agent_probe.py" 8107 qwen3.8-flash-next consistency3 2>&1 | sed 's/^/  /'
echo; echo "=== ② 多轮画像（4 轮 × 400 tok）==="
python3 "$REPO/qwen38_agent_probe.py" 8107 qwen3.8-flash-next multiturn 4 400 2>&1 | sed 's/^/  /'
echo; echo "=== ③ 单流回归（n=5）==="
python3 "$REPO/qwen38_agent_probe.py" 8107 qwen3.8-flash-next singlestream 2>&1 | sed 's/^/  /'
echo "=== done，服务保持运行 $(date -u +%FT%TZ) ==="
