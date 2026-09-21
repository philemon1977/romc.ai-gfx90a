#!/bin/bash
# QR C2+C3 的 A/B（无人值守）：等卡空 → A 臂基线（QR 关）→ 电池 → 停服 → B 臂（QR 开）→ 电池 → 停服 → 汇总
#
# 为什么需要起服：判据只有三个，全部要真服务——
#   ① 容器日志出现 QUICK_REDUCE（tp:0 的后端列表里确实选了它）
#   ② 事实召回 6/6 不掉（FP 模式应无损）
#   ③ TPS 同条件对拍（单流/并发 8/32）
#
# 规矩（每条都对应踩过的坑）：
#   · 绝不抢卡：起服前要求「无别的 api_server」且「八张 GCD 每张占用 <5 GiB」（与 launcher 硬门同阈值）；
#   · 只操作自己的容器名 glm53-int4，绝不 pkill -f（曾因自己的命令行匹配到模式而自杀）；
#   · 日志带时间戳、按臂分文件，不用"同分钟同名"。
set -u
ROOT=/home/qiba/ROCm.AI
L=/home/qiba/ai/models/ZhipuAI/launcher/glm53_int4w4a16_vllm_rocmnightly0918_32k_8121_mi250dx8.sh
PORT=${PORT:-8121}
MODEL=glm-5.3
BASE_IMG=rocm-ai/vllm:glm53-int4-gfx90a-0918
QR_IMG=rocm-ai/vllm:glm53-int4-gfx90a-0918-qr
TS=$(date +%Y%m%d_%H%M%S)
D=$ROOT/quark-int8/logs/qr_ab_$TS
mkdir -p "$D"
LOG=$D/run.log
MAXWAIT=${MAXWAIT:-28800}
READYWAIT=${READYWAIT:-2700}

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
vram_line(){ rocm-smi --showmeminfo vram 2>/dev/null | grep 'Used Memory' | awk '{printf "%.1f ", $NF/1073741824}'; }

gate(){
  pgrep -af 'vllm.entrypoints.openai.api_server' 2>/dev/null | grep -v -- "--port $PORT" | grep -q . && return 1
  local h f u
  for h in 0 1 2 3 4 5 6 7; do
    f=/sys/class/drm/card$((h+1))/device/mem_info_vram_used
    u=$(cat "$f" 2>/dev/null || echo 0)
    [ "$u" -gt $((5*1024*1024*1024)) ] && return 1
  done
  return 0
}

wait_free(){
  local t0 last=0 el
  t0=$(date +%s)
  while :; do
    if gate; then log "卡空闲（占用: $(vram_line)GiB），开始起服"; return 0; fi
    el=$(( $(date +%s) - t0 ))
    [ "$el" -ge "$MAXWAIT" ] && { log "等待超时 ${MAXWAIT}s，放弃本轮"; return 1; }
    if [ $(( el - last )) -ge 300 ]; then
      last=$el
      log "仍在等（已 ${el}s；占用: $(vram_line)GiB）"
    fi
    sleep 60
  done
}

ready(){
  local t0
  t0=$(date +%s)
  while :; do
    if python3 - "$PORT" <<'PY'
import sys, urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:%s/v1/models" % sys.argv[1], timeout=4).read()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
    then return 0; fi
    [ $(( $(date +%s) - t0 )) -ge "$READYWAIT" ] && return 1
    sleep 15
  done
}

battery(){
  local tag=$1 c n
  log "  [$tag] 事实召回"
  python3 "$ROOT/quark-int8/fact_recall_probe.py" "$PORT" "$MODEL" > "$D/${tag}_recall.txt" 2>&1
  tail -3 "$D/${tag}_recall.txt" | sed "s/^/  [$tag] /" | tee -a "$LOG"
  for c in 1 8 32; do
    n=32; [ "$c" = 1 ] && n=4; [ "$c" = 8 ] && n=16
    log "  [$tag] TPS conc=$c n=$n"
    python3 "$ROOT/quark-int8/qr_tps_probe.py" "$PORT" "$MODEL" "$c" "$n" 128 >> "$D/${tag}_tps.jsonl" 2>&1
    tail -1 "$D/${tag}_tps.jsonl" | sed "s/^/  [$tag] /" | tee -a "$LOG"
  done
}

stop_srv(){
  docker stop glm53-int4 >/dev/null 2>&1
  docker rm -f glm53-int4 >/dev/null 2>&1
  local i
  for i in $(seq 1 90); do gate && return 0; sleep 10; done
  return 0
}

arm(){
  local tag=$1 img=$2 qr=$3
  log "=== ${tag} 起服（镜像 ${img}，QR=${qr}）==="
  if [ "$qr" = on ]; then
    env IMAGE="$img" PORT="$PORT" \
        VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB=0 \
        VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=FP \
        VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0 \
        bash "$L" >> "$D/${tag}_serve.log" 2>&1
  else
    env IMAGE="$img" PORT="$PORT" bash "$L" >> "$D/${tag}_serve.log" 2>&1
  fi
  if ! ready; then
    log "!! ${tag} 起服/就绪失败，抓最后 40 行容器日志"
    docker logs --tail 40 glm53-int4 >> "$D/${tag}_serve.log" 2>&1
    stop_srv; return 1
  fi
  log "${tag} 就绪"
  docker logs glm53-int4 2>&1 | grep -m5 -E 'Using \[|QUICK_REDUCE|quick_reduce' > "$D/${tag}_qr_marker.txt" 2>&1 || true
  if [ -s "$D/${tag}_qr_marker.txt" ]; then
    log "  [$tag] 后端/QR 相关日志行:"
    head -5 "$D/${tag}_qr_marker.txt" | sed 's/^/    /' | tee -a "$LOG"
  else
    log "  [$tag] 日志里没有 Using [/QUICK_REDUCE 行"
  fi
  battery "$tag"
  log "${tag} 停服"
  stop_srv
}

wait_free || exit 1
# SKIP_A=1：跳过 stock 基线（074831 那轮已跑过，结果在 A_baseline_*）
[ "${SKIP_A:-0}" = 1 ] || arm A_baseline "$BASE_IMG" off
# A' = 同一个 -qr 镜像、三条 env 全不设 ⇒ 检验"打了补丁的代码本身是否中性"
[ "${SKIP_A2:-0}" = 1 ] || arm A2_qrimage_off "$QR_IMG" off
arm B_qr "$QR_IMG" on
{
  echo "# QR C2+C3 A/B 汇总（$TS）"
  echo
  echo "## A 臂基线（QR 关，镜像 $BASE_IMG）"
  tail -3 "$D/A_baseline_recall.txt" 2>/dev/null
  cat "$D/A_baseline_tps.jsonl" 2>/dev/null
  echo
  echo "## B 臂 QR 开（镜像 $QR_IMG + 三条 env）"
  tail -3 "$D/B_qr_recall.txt" 2>/dev/null
  cat "$D/B_qr_tps.jsonl" 2>/dev/null
  echo
  echo "## QR marker"
  echo "A: $(head -3 "$D/A_baseline_qr_marker.txt" 2>/dev/null)"
  echo "B: $(head -3 "$D/B_qr_qr_marker.txt" 2>/dev/null)"
} > "$D/summary.txt"
log "全部完成，汇总：$D/summary.txt"
echo "$D"
