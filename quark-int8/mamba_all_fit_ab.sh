#!/usr/bin/env bash
# A 方案第二轮：在**能装下的几何**下做 all vs align 的公平对照。
# 背景（2026-09-18 实测）：all 模式每个 block 留一份 mamba 状态 ⇒ maxlen=262144 时单请求要
# 16.47 GiB，而可用只有 5.95 GiB（超 2.77×）⇒ 引擎起不来（ValueError, 非代码错）。
# 线性缩放：maxlen=65536 ⇒ 4.12 GiB ✅。本脚本就以 65536 同几何对照 all vs align。
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int8w8a8attn_aiter_vllm_rocm72_mtp5_256k_8117_ornith_mi250dx8.sh
source "$REPO/_guard.sh"
exec > >(tee -a "$REPO/logs/mamba_all_fit_ab.log") 2>&1
echo "=== mamba all@64k vs align@64k  $(date -u +%FT%TZ)  MY_PORT=$MY_PORT ==="

measure_all () {
  local tag=$1
  echo "  --- V2 数值硬门（命中复用 vs 冷算）---"
  python3 "$REPO/reuse_consistency.py" 8127 "$tag" 64 1e-2 2>&1 | sed 's/^/    /'
  echo "  --- 长回答多轮（4 轮 × 1500 tok）---"
  python3 "$REPO/multiturn_probe.py" 8127 "$tag" 4 1500 2>&1 | sed 's/^/    /'
  echo "  --- 短回答多轮（4 轮 × 64 tok）---"
  python3 "$REPO/multiturn_probe.py" 8127 "$tag-short" 4 64 2>&1 | sed 's/^/    /'
  echo "  --- 单流 count（n=3）---"
  python3 "$REPO/measure_median.py" 8127 "$tag" 3 256 count 2>&1 | tail -1 | sed 's/^/    /'
  echo "  --- 长 prompt 冷/暖 TTFT ---"
  python3 "$REPO/prefix_probe.py" 8127 "$tag-pfx" 2>&1 | grep -aE "冷请求|暖请求|hits|queries|命中率" | sed 's/^/    /'
}

run_arm () {  # tag maxlen extra envx
  local tag=$1 maxlen=$2 extra=$3 envx=${4:-}
  echo; echo "######## ARM $tag  MAXLEN=$maxlen EXTRA='$extra' ALL=$envx  $(date -u +%H:%M:%S) ########"
  stop_mine; wait_released
  guard_no_other_servers || { echo "⛔ 有别的会话在服务，退出"; return 1; }
  guard_vram_free || return 1
  export SPEC=5 VLLM_EXTRA_ARGS="$extra" MAX_MODEL_LEN="$maxlen" MY_PORT=8127 PORT=8127
  export PID_FILE="/home/qiba/ai/logs/ornith397b-8127-dsh.pid"
  export LOG_FILE="/home/qiba/ai/logs/ornith397b/server-8127-dsh-$(date +%Y%m%d-%H%M).log"
  if [ -n "$envx" ]; then export VLLM_QWEN35_MAMBA_ALL=1; else unset VLLM_QWEN35_MAMBA_ALL; fi
  bash "$LAUNCH" || { echo "[$tag] ❌ LAUNCH FAILED"; return 1; }
  local t0=$(( $(date +%s) ))
  for i in $(seq 1 200); do curl -sf http://127.0.0.1:8127/health >/dev/null 2>&1 && break; sleep 5; done
  if ! curl -sf http://127.0.0.1:8127/health >/dev/null 2>&1; then
    echo "[$tag] ❌ NOT READY（$(date -u +%H:%M:%S)）；关键报错："
    grep -aoE "(ValueError|RuntimeError|NotImplementedError|AssertionError): .{0,200}" "$LOG_FILE" | sort -u | head -3 | sed 's/^/    /'
    return 1
  fi
  echo "  ✅ ready $(( $(date +%s)-t0 ))s"
  grep -aoE "enable_prefix_caching=[A-Za-z]+|cache mode is set to '[a-z]+'|GPU KV cache size: [0-9,]+ tokens" "$LOG_FILE" | sort -u | sed 's/^/    /'
  measure_all "$tag"
}

run_arm all-64k   65536 "--enable-prefix-caching --mamba-cache-mode all" 1
run_arm align-64k 65536 "--enable-prefix-caching" 0

echo; echo "######## 收尾：恢复出厂态到 8117（262144，前缀缓存关）########"
stop_mine; wait_released
if pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
  echo "⛔ 还有 vLLM 进程（可能是别的会话）⇒ 不抢，收尾跳过"; exit 0
fi
unset VLLM_EXTRA_ARGS VLLM_QWEN35_MAMBA_ALL MAX_MODEL_LEN
PORT=8117 SPEC=5 bash "$LAUNCH" || echo "⚠️ 8117 起服失败"
for i in $(seq 1 240); do curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 && break; sleep 5; done
curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 && echo "✅ 8117 出厂态已恢复"
echo "=== done $(date -u +%FT%TZ) ==="
