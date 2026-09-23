#!/bin/bash
# 用可达上下文把 GLM-5.3 真正跑起来：32K 上下文 + cudagraph，做质量验证
set -u
R=/home/qiba/ROCm.AI/quark-int8
LOG=$R/logs/glm53_run32k_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== GLM-5.3 int4 @ 32K 起服与质量验证 $(date +%T) ==="
PORT=8121 MAX_MODEL_LEN=32768 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=4096 GPU_MEM_UTIL=0.97 \
  ENFORCE_EAGER=0 MAX_CUDAGRAPH_CAPTURE_SIZE=128 \
  bash /home/qiba/ai/models/ZhipuAI/launcher/glm53_int4w4a16_vllm_rocmnightly0918_32k_8121_mi250dx8.sh || exit 1
L=$(ls -t /home/qiba/ai/logs/glm53/server-8121-*.log | head -1); echo "SERVER_LOG=$L"
for i in $(seq 1 180); do
  grep -qa "Application startup complete" "$L" 2>/dev/null && { echo "READY $(date +%T)"; break; }
  grep -qaE "ValueError|AttributeError|RuntimeError" "$L" 2>/dev/null && { echo "FAILED $(date +%T)"; grep -aE "ValueError|AttributeError|RuntimeError" "$L" | tail -3 | cut -c1-200; break; }
  docker ps -q -f name=glm53-int4 | grep -q . || { echo "CONTAINER GONE"; tail -15 "$L" | cut -c1-200; break; }
  sleep 20
done
if grep -qa "Application startup complete" "$L" 2>/dev/null; then
  echo "== 取证 =="
  grep -aE "TritonW4A16LinearKernel|WNA16 MoE backend|kernel module = mi250|Model loading took|GPU KV cache size|maximum concurrency|Capturing CUDA graphs \((PIECEWISE|FULL)\)" "$L" | tr "\r" "\n" | tail -8 | cut -c1-175
  echo "== 事实召回 =="; python3 $R/fact_recall_probe.py 8121 glm-5.3
  echo "== GSM8K 6 题（贪婪） =="
  python3 - <<PY
import json, re, urllib.request
Q = [("Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "72"),
     ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "10"),
     ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", "624"),
     ("Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. In the morning she gives 15 cups, in the afternoon 25 cups, for a flock of 20 chickens. How many cups in the final meal?", "20"),
     ("Kylar buys 16 glasses at $5 each, but every second glass costs 60% of the price. How much does he pay?", "64"),
     ("Betty needs $100. She has half. Parents give $15, grandparents twice that. How much more does she need?", "35")]
ok = 0
for q, gold in Q:
    d = json.dumps({"model": "glm-5.3", "prompt": "Question: " + q + "\nAnswer:", "max_tokens": 256, "temperature": 0}).encode()
    r = urllib.request.Request("http://127.0.0.1:8121/v1/completions", d, {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(r, timeout=1800))["choices"][0]["text"]
    hit = bool(re.search(r"(?<![0-9.])" + gold + r"(?![0-9])", t)); ok += hit
    print("  %s gold=%-4s -> %r" % ("OK" if hit else "XX", gold, t[:130]))
print("GSM8K 命中 %d/6" % ok)
PY
fi
echo "== 停服 $(date +%T) =="; docker rm -f glm53-int4 >/dev/null 2>&1; sleep 8
rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{printf "%.1f ", $NF/1073741824} END{print "GiB"}'
echo "RUN32K_DONE $(date +%T)"
