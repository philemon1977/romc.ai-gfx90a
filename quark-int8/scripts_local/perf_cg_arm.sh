#!/bin/bash
# 性能基线臂：engram 关闭（DSV41_ENG_SKIP=1）+ cudagraph + 修复后的 indexer 内核
set -u
BASH_SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$BASH_SOURCE_DIR/arm_teardown_guard.sh"
bash "$BASH_SOURCE_DIR/arm_preflight.sh" "$$"
PIDF=/home/qiba/ai/logs/dsv41ctint4-8119.pid
LOG=/home/qiba/ROCm.AI/quark-int8/logs/perf_cg_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== cudagraph 基线臂（engram off） $(date +%T) ==="
export MODEL_PATH=${MODEL_PATH:-/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4-opt-allbf16}
unset DSV41_ENG_HOST DSV41_ENG_HOST_PREALLOC 2>/dev/null || true
# ★ 不要 unset DSV41_IDX_AITER_KERNEL：它是本臂的关键开关（2026-09-20 踩坑：
#   这里 unset 掉 + 启动器白名单漏项，两个原因叠加 ⇒ 内核从未启用、白等 12 分钟）
export DSV41_IDX_AITER_KERNEL=${DSV41_IDX_AITER_KERNEL:-0}
EXPECT_IDX="$DSV41_IDX_AITER_KERNEL"   # ★ 在任何 unset 之前取期望值，自检才有意义
export DSV41_ENG_SKIP=${DSV41_ENG_SKIP:-1}   # 默认跳过；设 0 则 engram 正常驻显存
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.95} MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-8} MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
export ENFORCE_EAGER=${ENFORCE_EAGER:-0} MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-256}
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}
ISL_LIST=${ISL_LIST:-128,1024,4096}
T0=$(date +%s)
D=/home/qiba/ai/models/deepseek-ai/launcher/deepseek_v4.1_flash_ctint4w4a16_vllm_rocmnightly0918_64k_8119_dsv41_mi250dx8.sh
bash "$D" > /tmp/perf_cg_launch.log 2>&1
grep -E "已后台启动|❌" /tmp/perf_cg_launch.log | head -3
sleep 20
# ★ 起臂后 env 自检：关键开关必须真的进了容器（2026-09-20 三次踩坑：白名单漏项 ⇒ 静默走错路径）
echo "=== 容器内关键 env 自检 ==="
docker exec dsv41-ct-int4 bash -c 'env | grep -E "^(DSV41_|MI250_MOE_GEMV=|VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=)" | sort' 2>/dev/null || echo "  (自检失败：容器未就绪)"
GOT_IDX=$(docker exec dsv41-ct-int4 bash -c 'echo $DSV41_IDX_AITER_KERNEL' 2>/dev/null)
echo "  期望 DSV41_IDX_AITER_KERNEL=$EXPECT_IDX 实际=$GOT_IDX"
[ "$EXPECT_IDX" != "$GOT_IDX" ] && { echo "❌ env 未透传，立即中止（避免白等一轮）"; P=$(cat /home/qiba/ai/logs/dsv41ctint4-8119.pid 2>/dev/null); [ -n "$P" ] && kill -TERM -$P 2>/dev/null; exit 2; }
L=$(readlink -f /home/qiba/ai/logs/dsv41ctint4/server-8119.current)
echo "SERVER_LOG=$L"
for i in $(seq 1 120); do
  grep -qa "Application startup complete" "$L" && { echo "READY $(date +%T) 总耗时 $(( $(date +%s) - T0 ))s"; break; }
  grep -qaE "hipErrorStreamCaptureUnsupported|No available memory|EngineCore failed|WorkerProc initialization failed|Worker proc .* died unexpectedly" "$L" && { echo "FAILED $(date +%T)"; grep -aE "hipErrorStream|Available KV|No available memory" "$L" | tail -3 | cut -c1-190; break; }
  docker ps -q -f name=dsv41-ct-int4 | grep -q . || { echo "CONTAINER GONE $(date +%T)"; break; }
  sleep 20
done
if grep -qa "Application startup complete" "$L"; then
  echo "=== 证据 ==="
  grep -aE "Model loading took|Available KV|GPU KV cache size|Capturing CUDA graphs \(FULL\)|Breakable CUDA graph" "$L" | tr "\r" "\n" | tail -6 | cut -c1-175
  echo "=== 事实召回 ==="; python3 /home/qiba/ROCm.AI/quark-int8/fact_recall_probe.py 8119 /models
  echo "=== TTFT / TPS ==="
  python3 /home/qiba/ROCm.AI/quark-int8/perf_bench.py --port 8119 --isl "$ISL_LIST" \
     --max-tokens 16 --conc 1,4,8 --conc-isl 512 --conc-tokens 64 --label cg \
     --out /home/qiba/ROCm.AI/quark-int8/logs/perf_cg_$(date +%m%d_%H%M).json
fi
echo "=== 收尾（只在容器属本轮时） ==="
if owns_container; then
  [ -f "$PIDF" ] && kill -TERM -"$(cat "$PIDF")" 2>/dev/null; sleep 15; docker rm -f dsv41-ct-int4 >/dev/null 2>&1
else echo "  容器不属于本轮，跳过"; fi
rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{printf "%.2f ", $NF/1073741824} END{print "GiB"}'
echo "PERF_CG_DONE $(date +%T)"