#!/bin/bash
# 单臂"配置栈"探针：用给定的一组 env 起服 → 事实召回 + TPS(1/8/32) → 停服 → 追加到结果表。
#
#   用法：bash stack_probe.sh <臂名> [KEY=VALUE ...]
#   例：  bash stack_probe.sh MAX MI250_SPARSE_SPLITK=8 VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=FP
#
# 与 qr_ab_watch.sh 的区别（都是今天踩出来的）：
#   · fail-fast：装载期间每轮都查容器 State，一旦 exited 立刻退出并留日志（上一轮 B 臂
#     秒死却白等 45 分钟 ready 超时，就是这么烧掉一个空窗的）；
#   · 臂名与 env 由参数给，便于逐臂累加进同一张结果表；
#   · 沿用同一把尺子（qr_tps_probe.py 请求级吞吐 / fact_recall_probe.py 6 条），
#     因此只有**同表内**的臂可以互比，跨报告数字不可混用。
# 铁律：等卡空才起（无别的 api_server + 八张 GCD 各 <5 GiB）；只操作自己的容器名。
set -u
ROOT=/home/qiba/ROCm.AI
L=/home/qiba/ai/models/ZhipuAI/launcher/glm53_int4w4a16_vllm_rocmnightly0918_32k_8121_mi250dx8.sh
QR_IMG=rocm-ai/vllm:glm53-int4-gfx90a-0918-qr
PORT=${PORT:-8121}
MODEL=glm-5.3
TAG=${1:?用法: stack_probe.sh <臂名> [KEY=VALUE ...]}
shift || exit 2
EXTRA_ENVS=("$@")
TS=$(date +%Y%m%d_%H%M%S)
D=$ROOT/quark-int8/logs/stack
mkdir -p "$D"
LOG=$D/${TAG}_${TS}.log
RESULTS=$D/results.jsonl
READYWAIT=${READYWAIT:-1800}

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
vram(){ rocm-smi --showmeminfo vram 2>/dev/null | grep 'Used Memory' | awk '{printf "%.2f ", $NF/1073741824}'; }

# 门逻辑不在这里，统一用单测过的 quark-int8/gpu_gate.sh（这里的两份 bug 见该文件注释）
source "$ROOT/quark-int8/gpu_gate.sh"

log "=== 臂 ${TAG}：等卡（当前 $(vram_line)GiB）==="
if ! wait_free "$PORT" "${MAXWAIT:-7200}"; then log "等卡超时，放弃"; exit 1; fi
log "卡空闲，起服"

log "env: ${EXTRA_ENVS[*]:-（无额外 env）}"
if [ "${SKIP_BOOT:-0}" = 1 ]; then
  log "SKIP_BOOT=1：不起服，只检验就绪/判死逻辑（单测用）"
else
  env IMAGE="$QR_IMG" PORT="$PORT" ${EXTRA_ENVS[@]+"${EXTRA_ENVS[@]}"} bash "$L" >> "$LOG" 2>&1
fi

log "等就绪（最多 ${READYWAIT}s，容器真死了才撤）"
t0=$(date +%s); ready=0; dead=0; deadn=0; missn=0
while :; do
  # 判死要连续两次，且把「容器不存在」和「docker CLI 抖动」分开：一次误判就会杀掉
  # 一个正在装载 402 GB 的健康服务（今天真这么烧掉一个窗口，别再来一次）。
  if [ -z "$(docker ps -aq --filter name=^glm53-int4$ 2>/dev/null)" ]; then
    missn=$((missn+1))
    if [ "$missn" -ge 2 ]; then dead=1; break; fi
  else
    missn=0
    st=$(docker inspect -f '{{.State.Status}}' glm53-int4 2>/dev/null || echo unknown)
    if [ "$st" != "running" ]; then
      deadn=$((deadn+1))
      if [ "$deadn" -ge 2 ]; then dead=1; break; fi
    else
      deadn=0
    fi
  fi
  if python3 - "$PORT" <<'PY'
import sys, urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:%s/v1/models" % sys.argv[1], timeout=4).read(); sys.exit(0)
except Exception:
    sys.exit(1)
PY
  then ready=1; break; fi
  [ $(( $(date +%s) - t0 )) -ge "$READYWAIT" ] && break
  sleep 15
done
if [ "$dead" = 1 ]; then
  # 先存盘再删：上一臂先 rm 后抓日志，现场直接没了（只能去捞 launcher 的 tail 文件）
  CRASH=$D/${TAG}_${TS}_crash.log
  docker logs glm53-int4 > "$CRASH" 2>&1 || true
  if [ ! -s "$CRASH" ]; then
    F=$(ls -t /home/qiba/ai/logs/glm53/server-$PORT-*.log 2>/dev/null | head -1)
    if [ -n "$F" ]; then cp "$F" "$CRASH"; echo "  (容器已不在，改用 launcher 的 tail 日志 $F)"; fi
  fi
  log "!! 容器已退出，整份日志：$CRASH"
  grep -E 'Error|Exception|assert|Invalid|raise|ValueError|RuntimeError' "$CRASH" 2>/dev/null | head -6 | tee -a "$LOG"
  docker rm -f glm53-int4 >/dev/null 2>&1; exit 1
fi
if [ "$ready" != 1 ]; then
  log "!! 就绪超时 ${READYWAIT}s"
  docker logs --tail 40 glm53-int4 2>&1 | tail -12 | tee -a "$LOG"
  docker rm -f glm53-int4 >/dev/null 2>&1; exit 1
fi
log "就绪"
docker logs glm53-int4 2>&1 | grep -m6 -E 'Using \[|QUICK_REDUCE|SPLITK|split-K|WNA16 MoE backend' | sed 's/^/  /' | tee -a "$LOG"
python3 "$ROOT/quark-int8/fact_recall_probe.py" "$PORT" "$MODEL" > "$D/${TAG}_${TS}_recall.txt" 2>&1
tail -1 "$D/${TAG}_${TS}_recall.txt" | sed 's/^/  /' | tee -a "$LOG"
for c in 1 8 32; do
  n=32; [ "$c" = 1 ] && n=4; [ "$c" = 8 ] && n=16
  line=$(python3 "$ROOT/quark-int8/qr_tps_probe.py" "$PORT" "$MODEL" "$c" "$n" 128 2>&1 | tail -1)
  # 先前这里用 bash -c 传 "$line" 是错的：$line 没 export，子 shell 里是空串，
  # 结果表因此写进三行空记录（已清理）。改成 stdin 管道。
  printf "%s" "$line" | ENVS="${EXTRA_ENVS[*]:-}" ARM="$TAG" TSX="$TS" python3 -c '
import json, os, sys
raw = sys.stdin.read().strip()
try:
    d = json.loads(raw)
except Exception:
    d = {"raw": raw[:200], "parse_error": True}
d["arm"] = os.environ["ARM"]; d["ts"] = os.environ["TSX"]
d["envs"] = os.environ.get("ENVS", "")
print(json.dumps(d, ensure_ascii=False))
' >> "$RESULTS"
  echo "  conc=$c $line" | tee -a "$LOG"
done
log "停服"
docker stop glm53-int4 >/dev/null 2>&1; docker rm -f glm53-int4 >/dev/null 2>&1
for i in $(seq 1 90); do gate "$PORT" >/dev/null 2>&1 && break; sleep 10; done
log "完成：$LOG"
