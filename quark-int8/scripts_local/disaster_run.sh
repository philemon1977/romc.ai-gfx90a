#!/bin/bash
# Desc: 8128 臂就绪后的一键判据跑：invar（报表 vs 前向）→ rate → phase → churn → probe_nll 留档
# 用法: bash disaster_run.sh <server-log>
set -uo pipefail
L="${1:?需要 server 日志路径}"
T=/home/qiba/ai/tools/probe_disaster.py
U=http://127.0.0.1:8128/v1
M=glm53flash-int8
OUT=/home/qiba/ai/logs/glm53flash-0918/disaster-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
echo "输出目录 $OUT"
for i in $(seq 1 120); do
  grep -qa 'Application startup complete' "$L" 2>/dev/null && { echo "ready after $((i*10))s"; break; }
  grep -qaE 'exited gracefully|EngineDead' "$L" 2>/dev/null && { echo '引擎死了，不跑'; tail -5 "$L"; exit 1; }
  sleep 10
done
grep -qa 'Application startup complete' "$L" || { echo '等 20 分钟仍未就绪'; tail -4 "$L"; exit 1; }

echo; echo '=== 0) invar：新尺子首次对活服务端跑，先看它自己的口径警告 ==='
python3 $T invar --url $U --model $M --text zh1 2>&1 | tee "$OUT/00-invar-zh1.log"
python3 $T invar --url $U --model $M --text en1 2>&1 | tee "$OUT/01-invar-en1.log"

echo; echo '=== 1) rate：绝对位置 + 全周期扫描（每段 16 次）==='
python3 $T rate --url $U --model $M --runs 16 --texts zh1,zh2,en1,seq --out "$OUT/10-rate.json" 2>&1 | tee "$OUT/10-rate.log"

echo; echo '=== 2) phase：前插 0..4 个词，看灾难位跟内容还是跟绝对下标 ==='
python3 $T phase --url $U --model $M --runs 12 --shifts 5 --texts zh1 --out "$OUT/11-phase.json" 2>&1 | tee "$OUT/11-phase.log"

echo; echo '=== 3) churn：两次之间灌长请求改变 KV block 分配 ==='
python3 $T churn --url $U --model $M --runs 8 --texts zh1 --out "$OUT/12-churn.json" 2>&1 | tee "$OUT/12-churn.log"

echo; echo '=== 4) probe_nll 留档（同一次 boot 的 NLL 量级）==='
python3 /home/qiba/ai/tools/probe_nll.py --url $U --model $M 2>&1 | tee "$OUT/20-probe-nll.log"

echo; echo '=== 汇总 ==='
grep -h '可疑位置\|含灾难位\|判读\|invar ' "$OUT"/*.log | head -20
echo "DONE $OUT"