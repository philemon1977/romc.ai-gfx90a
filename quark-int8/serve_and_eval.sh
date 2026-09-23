#!/bin/bash
# 起服 → 等就绪 → 跑 A/B 质量评测 → 停服 → 等显存释放。整轮无人值守。
#
# 用法： LABEL=old MODEL_PATH=/path/to/model bash serve_and_eval.sh
#
# 设计要点（每条都对应踩过的坑）：
#   1) 只操作自己的 PID 文件与容器名，绝不 pkill -f（曾因自己的命令行匹配到模式而自杀）；
#      停服一律走 launcher 的 PID 文件 / docker stop <自己的容器名>。
#   2) 起服前硬门：八张卡都必须有 ≥58 GiB 空闲，否则等待（绝不抢卡、绝不腾地方）。
#   3) 评测在**宿主**上跑（只用 stdlib urllib，不需要 torch），GSM8K 路径必须传宿主路径。
#   4) 日志按 LABEL 与时间戳命名，避免"同分钟同名"互相覆盖。
set -u
LABEL=${LABEL:?需要 LABEL}
MODEL_PATH=${MODEL_PATH:-/home/qiba/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4}
# ★★ 关键：A/B 两臂统一把上下文降到 65536。
#   依据（旧模型实测）：预算 0.97×64.0=62.08 GiB/rank；权重 50.2 + KV 1.97 ⇒ 其它开销≈9.9。
#   新模型权重 +1.14 GiB/rank（共享专家与注意力升 bf16）⇒ KV 只剩 ≈0.84 GiB
#   ⇒ KV token ≈ 595,565×(0.84/1.97) ≈ **253,900 < 262,144** ⇒ vLLM 会**直接拒绝启动** ✗
#   降到 65536 对本判定是中性的：评测最长 prompt ~600 token、两臂同参、
#   且 YaRN 的 original_max=65536/factor=16 与 max-model-len 无关 ⇒ 不影响"是否恢复"的结论 ✓
#   （256K 交付已在旧 checkpoint 上实测达成，见转换记录 §4.13。）
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
PORT=${PORT:-8119}
CT=${CT:-dsv41-ct-int4}
REPO=/home/qiba/ROCm.AI/quark-int8
L=/home/qiba/ai/models/deepseek-ai/launcher/deepseek_v4.1_flash_ctint4w4a16_vllm_rocmnightly_256k_8119_dsv41_mi250dx8.sh
TS=$(date +%m%d_%H%M)
LOG=$REPO/logs/serve_${LABEL}_${TS}.log
SRVLOGDIR=/home/qiba/ai/logs/dsv41ctint4

min_free() {   # 八卡中最小的空闲 MiB
  local m=999999 v
  for d in 0 1 2 3 4 5 6 7; do
    v=$(rocm-smi --showmeminfo vram -d "$d" 2>/dev/null | awk '/Used Memory/{print $NF; exit}')
    if [ -n "${v:-}" ]; then v=$(( (68702699520 - v) / 1048576 )); else v=0; fi
    [ "$v" -lt "$m" ] && m=$v
  done
  echo "$m"
}

echo "=== $(date +%T) [$LABEL] 起服前硬门：八卡最小空闲需 ≥58 GiB ==="
for i in $(seq 1 120); do
  m=$(min_free); echo "  t=$((i*10))s 最小空闲=${m} MiB"
  if [ "$m" -ge 59392 ]; then break; fi
  sleep 10
done
if [ "$m" -lt 59392 ]; then echo "❌ 显存不足（${m} MiB），放弃本轮，不腾地方"; exit 1; fi

echo "=== $(date +%T) [$LABEL] 启动服务 (MODEL_PATH=$MODEL_PATH) → $LOG ==="
# ★ 顺带打开逐层健康画像（DSV41_PROFILE=1）：这样**一次起服同时产出**
#   ① 质量 A/B 结论 ② 新模型 40 层的 rms_stream/改动量画像。
#   若新模型仍退化，画像直接告诉我们"是哪一层不干活"，省掉又一整轮 15 分钟的起服。
#   刻意**不**开 DSV41_DUMP_DEBUG：dump 落在容器 /tmp，必须在停服前 docker cp 才拿得到，
#   时序没卡准就整批丢数据；而画像写宿主日志，容器停了也留得住 ✓。
MODEL_PATH="$MODEL_PATH" MAX_MODEL_LEN="$MAX_MODEL_LEN" DSV41_PROFILE=1 DSV41_DUMP_DEBUG=1 \
  setsid nohup bash "$L" > "$LOG" 2>&1 &
LP=$!
echo "launcher pid=$LP"

echo "=== $(date +%T) 等就绪（最多 40 min）==="
ok=0
for i in $(seq 1 160); do
  sleep 15
  if curl -s -m 3 "http://127.0.0.1:${PORT}/v1/models" | grep -q '"id"'; then ok=1; break; fi
  # 快速失败：`setsid` 会立刻退出，故 `kill -0 $LP` 几乎恒为假、原写法实际检测不到失败，
  # 会白等满 40 分钟 ✗ 改为直接扫 launcher 日志里的致命错误行 ✓
  if grep -qaE "Traceback \(most recent call last\)|EngineMissingCapabilitiesError|No available memory|larger than the maximum number of tokens|ValueError: Call with " "$LOG" 2>/dev/null; then
    echo "❌ $(date +%T) 起服日志出现致命错误，快速失败（不再白等）"; break
  fi
  [ $((i % 8)) -eq 0 ] && echo "  ...等就绪 $((i*15))s"
done
if [ "$ok" -ne 1 ]; then
  echo "❌ 服务未就绪；日志尾部："; tail -25 "$LOG"
  echo "--- 服务日志尾部 ---"; tail -25 "$SRVLOGDIR/server-8119.current" 2>/dev/null
  exit 1
fi
echo "✅ $(date +%T) 服务就绪"

echo "=== $(date +%T) [$LABEL] 跑质量评测 ==="
python3 "$REPO/eval_quality_ab.py" --port "$PORT" --label "$LABEL" \
  --n-gsm8k "${NGSM8K:-8}" \
  --gsm8k "$REPO/refs/gsm8k_sample.jsonl" \
  --out "$REPO/logs/quality_${LABEL}_${TS}.json" 2>&1 | tail -40

# ★ 停服**之前**把容器内 dump 取出来：DUMP 文件落在容器 /tmp，容器一停就没了 ✗
#   （这是本会话踩过的时序坑，故显式前置。）layer2 是 Full 模式层，正是 H2（CSA2 压缩路径）
#   端到端对拍需要的输入。失败不影响评测结论，只影响 H2 素材。
echo "=== $(date +%T) [$LABEL] 取回探针 dump ==="
mkdir -p "$REPO/dumps/$LABEL"
timeout 60 docker cp "$CT:/tmp/." "$REPO/dumps/$LABEL/" 2>&1 | tail -1
ls -la "$REPO/dumps/$LABEL"/dsv41_*.pt 2>/dev/null | awk '{printf "  %s %.1f MB\n", $NF, $5/1048576}' | head -10

echo "=== $(date +%T) [$LABEL] 停服 ==="
if [ -f /home/qiba/ai/logs/dsv41ctint4-8119.pid ]; then
  PG=$(cat /home/qiba/ai/logs/dsv41ctint4-8119.pid)
  kill -TERM -"$PG" 2>/dev/null && echo "已 TERM 进程组 $PG"
fi
# ★ 停服前核对：只停"挂载的正是本 LABEL 模型"的那个容器。
#   容器名是全局的——若同名容器其实属于别的会话（或别的模型），绝不能停（CLAUDE.md §1）。
if docker ps --format '{{.Names}}' | grep -qx "$CT"; then
  MNT=$(timeout 20 docker inspect "$CT" --format '{{range .Mounts}}{{.Source}} {{end}}' 2>/dev/null)
  # docker 可能返回**符号链接解析后**的路径 ⇒ 字符串相等会误判"不是我的容器"而漏停，
  # 从而占住显存拖垮下一臂 ✗ 两侧都归一化后再比 ✓
  MRP=$(readlink -f "$MODEL_PATH" 2>/dev/null || echo "$MODEL_PATH")
  hit=0
  for one in $MNT; do
    [ "$(readlink -f "$one" 2>/dev/null || echo "$one")" = "$MRP" ] && hit=1
  done
  if [ "$hit" -eq 1 ]; then
    echo "  容器 $CT 挂载的正是本模型（归一化后匹配），停止它"; timeout 90 docker stop -t 30 "$CT" 2>&1 | tail -1
  else
    echo "  ⚠️ 容器 $CT 挂载($MNT)与本模型($MRP)不符 ⇒ **不碰它**，只走自己的 PID 文件"
  fi
else
  echo "  容器 $CT 已不在（可能已被 --rm 清理）"
fi
echo "=== 等显存释放 ==="
freed=0
for i in $(seq 1 90); do
  sleep 10; m=$(min_free)
  [ "$m" -ge 59392 ] && { echo "✅ $(date +%T) 显存已释放（${m} MiB）"; freed=1; break; }
  [ $((i % 6)) -eq 0 ] && echo "  t=$((i*10))s 最小空闲=${m} MiB"
done
if [ "$freed" -ne 1 ]; then
  echo "❌ 等 15 分钟仍有卡不足 58 GiB（最小 ${m} MiB）⇒ 放弃后续，不腾地方、不抢卡"
  exit 1
fi
echo "=== $(date +%T) [$LABEL] 完毕 ==="
