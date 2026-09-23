#!/bin/bash
# 单变量 A/B 起服 + 质量冒烟。用法：
#   ENG_OFF=1 GEMV_OFF=1 TAG=engoff_mygemvoff ./run_cycle_ab.sh
# 说明：DSV41_ENG_OFF=1 旁路 engram 注入（common_engram.py 早退）；
#       MI250_MOE_GEMV=0 用上游 Triton WNA16 MoE（不加载我们的 GEMV 内核）。
TAG="${TAG:-ab}"
ENG_OFF="${ENG_OFF:-0}"
GEMV_OFF="${GEMV_OFF:-0}"
EXTRA=""
[ "$ENG_OFF" = "1" ] && EXTRA="$EXTRA DSV41_ENG_OFF=1"
[ "$GEMV_OFF" = "1" ] && EXTRA="$EXTRA MI250_MOE_GEMV=0"
PIDF=/home/qiba/ai/logs/dsv41ctint4-8119.pid
[ -f "$PIDF" ] && { kill -TERM -"$(cat $PIDF)" 2>/dev/null; sleep 10; }
docker rm -f dsv41-ct-int4 >/dev/null 2>&1
LOG=/home/qiba/ROCm.AI/quark-int8/logs/bench_${TAG}.log
echo "[cycle $TAG] env:$EXTRA  日志 $LOG"
setsid nohup bash -c "PORT=8119 OUT=48 DSV41_LOGIT_DEBUG=1 $EXTRA bash /home/qiba/ROCm.AI/quark-int8/bench_serve.sh > $LOG 2>&1" &
S=$(readlink -f /home/qiba/ai/logs/dsv41ctint4/server-8119.current 2>/dev/null)
for i in $(seq 1 60); do curl -s -m 3 http://127.0.0.1:8119/health >/dev/null 2>&1 && { echo "[$TAG] ✅ READY $(date +%T)"; break; }; sleep 20; done
python3 - <<'PY'
import json,urllib.request
CASES=[("常识","The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is"),
       ("知识","The first president of the United States was"),
       ("算术","1+1=2, 2+2=4, 3+3="),
       ("代码","def add(a, b):\n    return"),
       ("中文","北京是中国的首都，东京是日本的首都，巴黎是")]
for n,p in CASES:
    try:
        r=urllib.request.Request("http://127.0.0.1:8119/v1/completions",
          data=json.dumps({"model":"/models","prompt":p,"max_tokens":16,"temperature":0}).encode(),
          headers={"Content-Type":"application/json"})
        d=json.loads(urllib.request.urlopen(r,timeout=400).read())
        print(f"  [{n}] {d['choices'][0]['text']!r}")
    except Exception as e: print(f"  [{n}] 失败 {type(e).__name__} {str(e)[:90]}")
PY
echo "[cycle $TAG] 完成 $(date +%T)"
