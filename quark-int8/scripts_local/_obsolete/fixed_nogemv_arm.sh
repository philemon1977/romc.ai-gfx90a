#!/bin/bash
# 修复后仓 + 关闭我们的 GEMV 内核（走上游 WNA16 Triton MoE）：单变量
set -u
PIDF=/home/qiba/ai/logs/dsv41ctint4-8119.pid
LOG=/home/qiba/ROCm.AI/quark-int8/logs/fixed_nogemv_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== 修复后 + GEMV=0 臂 $(date +%T) ==="
export MODEL_PATH=/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4-opt-allbf16
export DSV41_ENG_SKIP=1 MI250_MOE_GEMV=0
export GPU_MEM_UTIL=0.97 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=1024
D=/home/qiba/ai/models/deepseek-ai/launcher/deepseek_v4.1_flash_ctint4w4a16_vllm_rocmnightly0918_64k_8119_dsv41_mi250dx8.sh
bash "$D" > /tmp/nogemv_launch.log 2>&1
grep -E "已后台启动|❌" /tmp/nogemv_launch.log | head -3
sleep 20
L=$(readlink -f /home/qiba/ai/logs/dsv41ctint4/server-8119.current)
echo "SERVER_LOG=$L"
for i in $(seq 1 80); do
  grep -qa "Application startup complete" "$L" && { echo "READY $(date +%T)"; break; }
  grep -qaE "No available memory|OutOfMemory|EngineCore failed" "$L" && { echo "FAILED"; tail -12 "$L" | cut -c1-190; exit 1; }
  docker ps -q -f name=dsv41-ct-int4 | grep -q . || { echo "CONTAINER GONE"; exit 1; }
  sleep 20
done
grep -aE "Model loading took|GPU KV cache size|WNA16 MoE backend|kernel module = mi250" "$L" | tail -4 | cut -c1-180
echo "=== 事实召回 ==="
python3 /home/qiba/ROCm.AI/quark-int8/fact_recall_probe.py 8119 /models
echo "=== GSM8K 前 6 题（贪婪） + 重复检测 ==="
python3 - <<'PY'
import json, urllib.request, re
Q = ["Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
     "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
     "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
     "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
     "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
     "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?"]
GOLD = ["72", "10", "35", "624", "20", "64"]
def gen(p, n=320):
    d = json.dumps({"model": "/models", "prompt": "Question: " + p + "\nAnswer:", "max_tokens": n, "temperature": 0.0}).encode()
    r = urllib.request.Request("http://127.0.0.1:8119/v1/completions", d, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))["choices"][0]["text"]
ok = 0
for q, g in zip(Q, GOLD):
    t = gen(q)
    hit = bool(re.search(r"(?<![0-9.])" + g + r"(?![0-9])", t))
    ok += hit
    print("  %s gold=%-4s -> %r" % ("✔" if hit else "✘", g, t[:170]))
print("GSM8K 命中 %d/6" % ok)
PY
echo "=== 速度（单流 decode t/s, 128 tok ×3） ==="
python3 - <<'PY'
import json, urllib.request, time
def run(n=128):
    d = json.dumps({"model": "/models", "prompt": "请用中文写一段关于长江的介绍：", "max_tokens": n, "temperature": 0.0, "stream": False}).encode()
    r = urllib.request.Request("http://127.0.0.1:8119/v1/completions", d, {"Content-Type": "application/json"})
    t0 = time.time(); out = json.load(urllib.request.urlopen(r, timeout=900)); dt = time.time() - t0
    return out["usage"]["completion_tokens"] / dt
for i in range(3):
    print("  run%d: %.2f tok/s" % (i + 1, run()))
PY
echo "=== 停服还卡 ==="
[ -f "$PIDF" ] && kill -TERM -"$(cat $PIDF)" 2>/dev/null
sleep 15; docker rm -f dsv41-ct-int4 >/dev/null 2>&1
rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{printf "%.2f ", $NF/1073741824} END{print "GiB"}'
echo "NOGEMV_ARM_DONE $(date +%T)"
