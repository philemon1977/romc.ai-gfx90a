#!/bin/bash
# 可用配置臂：engram 开（显存）+ eager + 上游 indexer 回退
set -u
BASH_SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$BASH_SOURCE_DIR/arm_teardown_guard.sh"
bash "$BASH_SOURCE_DIR/arm_preflight.sh" "$$"
PIDF=/home/qiba/ai/logs/dsv41ctint4-8119.pid
LOG=/home/qiba/ROCm.AI/quark-int8/logs/perf_eng_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== engram开+eager 臂 $(date +%T) ==="
# 默认用最朴素的 -engram4（注意力普通 int4，无"合并模块 bf16+ignore 通配"的特殊路径）
export MODEL_PATH=${MODEL_PATH:-/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4}
unset DSV41_ENG_HOST DSV41_ENG_HOST_PREALLOC 2>/dev/null || true
export DSV41_IDX_AITER_KERNEL=${DSV41_IDX_AITER_KERNEL:-0}   # 与 cg 脚本同款修正：不要 unset 关键开关
EXPECT_IDX="$DSV41_IDX_AITER_KERNEL"
export DSV41_ENG_SKIP=${DSV41_ENG_SKIP:-0}   # 1 = 跳过 engram（已验证可服务）；0 = 打开（当前会崩，待修）
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.95} MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-8} MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
export ENFORCE_EAGER=1 MAX_CUDAGRAPH_CAPTURE_SIZE=0
ISL_LIST=${ISL_LIST:-128,1024,4096}
T0=$(date +%s)
D=/home/qiba/ai/models/deepseek-ai/launcher/deepseek_v4.1_flash_ctint4w4a16_vllm_rocmnightly0918_64k_8119_dsv41_mi250dx8.sh
bash "$D" > /tmp/perf_eng_launch.log 2>&1
grep -E "已后台启动|❌" /tmp/perf_eng_launch.log | head -3
sleep 20
L=$(readlink -f /home/qiba/ai/logs/dsv41ctint4/server-8119.current)
echo "SERVER_LOG=$L"
for i in $(seq 1 120); do
  grep -qa "Application startup complete" "$L" && { echo "READY $(date +%T) 总耗时 $(( $(date +%s) - T0 ))s"; break; }
  grep -qaE "No available memory|EngineCore failed|WorkerProc initialization failed" "$L" && { echo "FAILED $(date +%T)"; grep -aE "Available KV|No available memory|cancelled" "$L" | tail -3 | cut -c1-190; break; }
  docker ps -q -f name=dsv41-ct-int4 | grep -q . || { echo "CONTAINER GONE $(date +%T)"; break; }
  sleep 20
done
if grep -qa "Application startup complete" "$L"; then
  echo "=== 证据 ==="
  grep -aE "Model loading took|Available KV|GPU KV cache size|sparse MLA attention" "$L" | tr "\r" "\n" | tail -5 | cut -c1-175
  echo "=== 事实召回 ==="; python3 /home/qiba/ROCm.AI/quark-int8/fact_recall_probe.py 8119 /models
  echo "=== TTFT / TPS ==="
  python3 /home/qiba/ROCm.AI/quark-int8/perf_bench.py --port 8119 --isl "$ISL_LIST" \
     --max-tokens 16 --conc 1,4,8 --conc-isl 512 --conc-tokens 64 --label eng \
     --out /home/qiba/ROCm.AI/quark-int8/logs/perf_eng_$(date +%m%d_%H%M).json
fi
if owns_container; then
  [ -f "$PIDF" ] && kill -TERM -"$(cat "$PIDF")" 2>/dev/null; sleep 15; docker rm -f dsv41-ct-int4 >/dev/null 2>&1
else echo "  容器不属于本轮，跳过收尾"; fi
rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{printf "%.2f ", $NF/1073741824} END{print "GiB"}'
echo "PERF_ENG_DONE $(date +%T)"