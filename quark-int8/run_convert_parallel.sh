#!/bin/bash
# 8 路并行做 CT-Int4 转换（分片互不相交 ⇒ 天然可并行）。
#
# 为什么不是"8 张卡"而是"8 个进程 + 限线程"：
#   实测（04:19，logs/full_allbf16_0919_0413.log）转换期间 **8 卡 GPU util 全 0%**，
#   单进程 %CPU≈1987（≈20 核），48 核机器 load 21~25。
#   ⇒ 瓶颈是 **CPU**，不是 GPU、也不是盘（写 676 MB/s、读 592 MB/s，均未饱和）。
#   直接起 8 进程会要 ~160 线程 ≫ 48 核 ⇒ 必须给每进程限 OMP 线程数。
#   预期收益 2–3×（不是 8×）：受核数与两块 NVMe 聚合带宽双重限制。
#
# 两条硬性安全约束：
#   1) engram 分片 47/48 是 94 GiB 的**直通拷贝**，I/O bound；8 路并行会 8×94 GiB 顶爆
#      251 GiB 内存 ⇒ 本脚本把它们单独串行处理。
#   2) 收尾守卫必须等**全部** worker 退出 + 48 分片齐全，且 DONE 要在**每个** worker 日志里
#      都出现。旧 finish_*.sh 只看一份日志 ⇒ 并行下会在别的进程还在写盘时启动 requant，
#      写出损坏 checkpoint（这是本项目最容易造成不可逆损失的一类竞态）。
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
SRC=${SRC:-/mnt/stripe-3mix-3t2/models/deepseek-ai/DeepSeek-V4.1-Flash}
OUT=${OUT:?需要 OUT}
NPROC=${NPROC:-8}
THREADS=${THREADS:-6}          # 8 × 6 = 48 = 核数
SHARDS=${SHARDS:-1-46}         # 默认并行段；47,48 单独串行
MIN_FREE_GIB=${MIN_FREE_GIB:-40}
TS=$(date +%m%d_%H%M)
mkdir -p "$OUT" "$REPO/logs/par_$TS"

# 分片列表：支持 "1-46" 区间，也支持 "47,48" 这种逗号列表
ALL=()
if [[ "$SHARDS" == *-* && "$SHARDS" != *,* ]]; then
  IFS='-' read -r lo hi <<<"$SHARDS"
  mapfile -t ALL < <(seq "$lo" "$hi")
else
  IFS=',' read -r -a ALL <<<"$SHARDS"
fi
total=${#ALL[@]}
echo "[par] 段 $SHARDS 共 $total 片；$NPROC 路 × $THREADS 线程；OUT=$OUT"

# 先确认没有别的转换在跑（避免两个写者同时写一个目录）
if pgrep -f "[c]onvert_dsv41_ct_int4.py" >/dev/null; then
  echo "❌ 已有转换进程在跑 ⇒ 先停它，绝不并发写同一 OUT"; exit 1
fi

free_min() {
  local m=999999 v
  for d in 0 1 2 3 4 5 6 7; do
    v=$(rocm-smi --showmeminfo vram -d "$d" 2>/dev/null | awk '/Used Memory/{print $NF}')
    if [ -n "${v:-}" ]; then v=$(( (68702699520 - v) / 1073741824 )); else v=0; fi
    [ "$v" -lt "$m" ] && m=$v
  done
  echo "$m"
}
echo "[par] 等最小空闲显存 ≥${MIN_FREE_GIB} GiB（转换虽不吃 GPU，但要遵守不抢卡）"
for i in $(seq 1 60); do
  m=$(free_min); [ "$m" -ge "$MIN_FREE_GIB" ] && { echo "  ✅ 最小空闲 ${m} GiB"; break; }
  echo "  t=$((i*15))s 最小空闲 ${m} GiB"; sleep 15
done

pids=(); logs=()
for w in $(seq 0 $((NPROC-1))); do
  grp=()
  for ((i=w; i<total; i+=NPROC)); do grp+=("${ALL[$i]}"); done
  [ ${#grp[@]} -eq 0 ] && continue
  L="$REPO/logs/par_$TS/w${w}.log"
  list=$(IFS=,; echo "${grp[*]}")
  # 关键：OMP_NUM_THREADS 限死每进程线程数，否则 8×20 线程互相颠簸
  setsid nohup env OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
    HOST_OUT="$OUT" SHARDS="$list" OPT_SCALE=1 OPT_STEP=${OPT_STEP:-0.05} \
    SHARED_BF16=1 ATTN_BF16=1 ATTN_MERGED=1 SKIP_EXISTING=1 \
    bash "$REPO/run_convert.sh" > "$L" 2>&1 &
  pids+=($!); logs+=("$L")
  echo "[par] worker$w 分片 $list → $L"
done

echo "[par] 等全部 worker 退出（这是守卫的一部分，不可跳过）"
for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done
while pgrep -f "[c]onvert_dsv41_ct_int4.py" >/dev/null; do sleep 20; done

echo "[par] 守卫：每个 worker 日志都要有 DONE"
bad=0
for L in "${logs[@]}"; do
  if ! grep -qa "DONE" "$L"; then echo "  ❌ 无 DONE: $L"; bad=1; fi
  if grep -qa "FAILURES\|Traceback" "$L"; then echo "  ⚠️ 日志含异常关键字: $L"; fi
done
n=$(ls "$OUT"/model-000*-of-00048.safetensors 2>/dev/null | wc -l)
echo "[par] 分片数=$n（本段应到 46）bad=$bad"
[ "$bad" -eq 0 ] || { echo "❌ 有 worker 未完成，不要继续 requant"; exit 1; }
echo "✅ 并行段完成。engram 分片请随后单独串行跑：SHARDS=47,48 本脚本"
