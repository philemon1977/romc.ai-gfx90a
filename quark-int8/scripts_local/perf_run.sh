#!/bin/bash
# Desc: 8128 臂就绪后的性能实测：speed_probe（流式真 TTFT）+ perf_bench（ignore_eos 口径）+ 负载下的质量复测
# 用法: bash perf_run.sh <server-log>
set -uo pipefail
L="${1:?需要 server 日志}"
AI=/home/qiba/ai
REPO=/home/qiba/ROCm.AI
M=glm53flash-int8
OUT=$AI/logs/glm53flash-0918/perf-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"; echo "输出 $OUT"
echo "KV/配置证据: $(grep -aoE 'GPU KV cache size: [0-9,]+ tokens, Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' $L | tail -1)"
for i in $(seq 1 130); do
  grep -qa 'Application startup complete' "$L" 2>/dev/null && { echo "READY after $((i*10))s"; break; }
  grep -qaE 'exited gracefully|EngineDead' "$L" 2>/dev/null && { echo '引擎死了'; tail -4 $L; exit 1; }
  sleep 10
done
grep -qa 'Application startup complete' "$L" || { echo '20 分钟未就绪'; tail -4 $L; exit 1; }
curl_s() { python3 - "$1" <<'PY'
import sys,urllib.request
try: print(urllib.request.urlopen(sys.argv[1],timeout=30).read().decode())
except Exception as e: print('metrics 不可用:',e)
PY
}
echo; echo '### metrics 快照（前）' > $OUT/00-metrics.txt
curl_s http://127.0.0.1:8128/metrics | grep -E 'kv_cache_usage|prefix_cache_hit|num_requests_running|num_requests_waiting|preempt' >> $OUT/00-metrics.txt

echo; echo '===== A) speed_probe：单流 TTFT/解码（流式真 TTFT，无 ignore_eos）====='
python3 $REPO/quark-int8/speed_probe.py --port 8128 --model $M --lens 1024,4096,16384 --gen 64 --conc 1,4,8 2>&1 | tee $OUT/10-speed_probe.log

echo; echo '===== B) speed_probe：稳态长解码（单流 gen=256）====='
python3 $REPO/quark-int8/speed_probe.py --port 8128 --model $M --lens 1024,4096 --gen 256 --conc '' 2>&1 | tee $OUT/11-speed_probe_long.log

echo; echo '===== C) perf_bench：TTFT 用 max_tokens=1 近似 + ignore_eos/min_tokens 满生成 ====='
python3 $REPO/quark-int8/perf_bench.py --port 8128 --model $M --isl 128,1024,4096 --max-tokens 64 --conc 1,4,8 --conc-isl 512 --conc-tokens 64 --label 8128-int8 --out $OUT/12-perf_bench.json 2>&1 | tee $OUT/12-perf_bench.log

echo; echo '===== D) 负载后再量质量（同一 boot，看是否随负载漂移）====='
python3 $AI/tools/probe_nll.py --url http://127.0.0.1:8128/v1 --model $M 2>&1 | tee $OUT/20-probe-nll-after.log
python3 $REPO/quark-int8/scripts_local/taskcheck.py --url http://127.0.0.1:8128/v1 --model $M --tag 8128-perf --needle-tokens 2048 2>&1 | tail -12 | tee $OUT/21-taskcheck.log

echo; echo '### metrics 快照（后）'
curl_s http://127.0.0.1:8128/metrics | grep -E 'kv_cache_usage|prefix_cache_hit|num_requests_running|num_requests_waiting|preempt' >> $OUT/00-metrics.txt
echo; echo '### 服务端异常扫描（本次 boot 全量）'
grep -acE 'Traceback|CUDA error|HIP error|out of memory|Preempting|aborted' $L | sed 's/^/  可疑行数=/'
grep -aE 'Preempting|out of memory|HIP error' $L | tail -3 | cut -c1-140
echo DONE $OUT