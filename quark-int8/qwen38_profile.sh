#!/usr/bin/env bash
# P4 诊断：rocprofv3 附着到 8107 的 worker，采一个 decode 窗口的 kernel trace，
# 然后按"workgroup 数 vs 104 CU"排序 —— 找出"每步大半个 GPU 闲置"的内核。
#
# 为什么是这个诊断：前序 docs/research-notes/aiter-cdna2-4-量化与GEMM.md §B4/§B5 已算过本模型
# decode 离访存地板 12–16×（地板仅占 TPOT 6–8%）⇒ 瓶颈是内核**占用/发射**，不是带宽；
# 而 grid ≪ 104（MI250X 每 GCD 的 CU 数）就是"闲置"的直接证据。
#
# 已知雷（已在本机防御性修好，备份 .orig-20260918）：
#   rocprofv3 会往子进程 stdout 打 "SPM is not supported on gfx90a …"，而 aiter JIT 的
#   get_hip_version() 直接把 stdout 当版本号 int() 解析 ⇒ ValueError 崩服。
#   修法：aiter/jit/utils/cpp_extension.py 用正则取 (\d+)\.(\d+)。
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
OUT="${OUT:-/tmp/8107_prof_$(date +%H%M)}"
DUR_MS="${DUR_MS:-25000}"
ROCPROF="${ROCPROF:-/opt/rocm-7.2.4/bin/rocprofv3}"
exec > >(tee -a "$REPO/logs/qwen38_profile.log") 2>&1

echo "=== 8107 kernel-grid 诊断 $(date -u +%FT%TZ)  out=$OUT dur=${DUR_MS}ms ==="
curl -sf http://127.0.0.1:8107/health >/dev/null || { echo "❌ 8107 未就绪，先起服"; exit 1; }

# 找 worker（GPU 干活的是 worker 进程；rank0 足够，8 rank 形状对称）
W=$(pgrep -f "VLLM::Worker_TP0" | head -1)
[ -n "$W" ] || W=$(pgrep -f "VLLM::Worker" | head -1)
[ -n "$W" ] || { echo "❌ 找不到 worker 进程"; exit 1; }
echo "  worker pid=$W（$(tr '\0' ' ' < /proc/$W/cmdline | cut -c1-60)…）"
mkdir -p "$OUT"

# 先发一个 warmup（丢弃），保证后面采到的是稳态 decode
python3 "$REPO/qwen38_probe.py" 8107 qwen3.8-flash-next 64 1 >/dev/null 2>&1 || true

echo "  启动 rocprofv3 attach …"
"$ROCPROF" --attach "$W" --kernel-trace --output-format csv \
  -d "$OUT" --attach-duration-msec "$DUR_MS" > "$OUT/rocprof.out" 2>&1 &
RP=$!
sleep 4
echo "  采集窗口内发 decode 请求（256 tok × 3）…"
python3 "$REPO/qwen38_probe.py" 8107 qwen3.8-flash-next 256 3 2>&1 | sed 's/^/    /'
wait $RP 2>/dev/null || true
echo "  rocprofv3 退出；产出："
ls -l "$OUT" | tail -6

CSVS=$(ls "$OUT"/*kernel*trace*.csv 2>/dev/null | head -3)
if [ -z "$CSVS" ]; then
  echo "  ⚠️ 没拿到 kernel trace CSV；看 $OUT/rocprof.out："
  tail -12 "$OUT/rocprof.out" | sed 's/^/    /'
  exit 1
fi
echo "=== 分析（workgroup 数 vs 104 CU）==="
python3 "$REPO/analyze_kernel_csv.py" $CSVS --top=25 --cu=104
echo "=== done $(date -u +%FT%TZ) ==="
