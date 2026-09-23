#!/bin/bash
# 修复后（-opt-allbf16 仓，专家 nibble 序已就地修复）单变量验证臂
# 与之前退化臂同配置，唯一变量 = 专家权重不再是「相邻对互换」
set -u
PIDF=/home/qiba/ai/logs/dsv41ctint4-8119.pid
LOG=/home/qiba/ROCm.AI/quark-int8/logs/fixed_allbf16_$(date +%m%d_%H%M).log
mkdir -p /home/qiba/ROCm.AI/quark-int8/logs
exec > >(tee -a "$LOG") 2>&1
echo "=== 修复后臂 $(date +%T) 日志 $LOG ==="

export MODEL_PATH=/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4-opt-allbf16
export DSV41_ENG_SKIP=1
export GPU_MEM_UTIL=0.97 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=1024
D=/home/qiba/ai/models/deepseek-ai/launcher/deepseek_v4.1_flash_ctint4w4a16_vllm_rocmnightly0918_64k_8119_dsv41_mi250dx8.sh
bash "$D" > /tmp/fixed_launch.log 2>&1
grep -E "已后台启动|❌" /tmp/fixed_launch.log | head -3
sleep 20
L=$(readlink -f /home/qiba/ai/logs/dsv41ctint4/server-8119.current)
echo "SERVER_LOG=$L"
for i in $(seq 1 80); do
  grep -qa "Application startup complete" "$L" && { echo "READY $(date +%T)"; break; }
  if grep -qaE "No available memory|OutOfMemory|EngineCore failed|Error in engine" "$L"; then
    echo "FAILED $(date +%T)"; grep -aE "Available KV|No available memory|OutOfMemory|Error" "$L" | grep -av site-packages | tail -4 | cut -c1-200
    [ -f "$PIDF" ] && kill -TERM -"$(cat $PIDF)" 2>/dev/null; exit 1
  fi
  docker ps -q -f name=dsv41-ct-int4 | grep -q . || { echo "CONTAINER GONE"; tail -20 "$L" | cut -c1-200; exit 1; }
  sleep 20
done
echo "=== 装载证据 ==="
grep -aE "Model loading took|GPU KV cache size|WNA16 MoE backend|kernel module = mi250" "$L" | tail -4 | cut -c1-180
echo "=== 事实召回 ==="
python3 /home/qiba/ROCm.AI/quark-int8/fact_recall_probe.py 8119 /models
echo "=== 长生成（复读检测，256 tok） ==="
python3 - <<'PY'
import json, urllib.request
def q(p, n=160, t=0.0):
    d = json.dumps({"model": "/models", "prompt": p, "max_tokens": n, "temperature": t}).encode()
    r = urllib.request.Request("http://127.0.0.1:8119/v1/completions", d, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))["choices"][0]["text"]
for p, n in [("What is 17*19? Answer: 17*19 =", 48),
             ("Question: Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:", 220),
             ("中国的首都是北京。法国的首都是", 120),
             ("请用三句话介绍量子纠缠。", 200)]:
    txt = q(p, n)
    # 复读检测：最长重复后缀占比
    rep = 0
    for L in range(4, len(txt)//2 + 1):
        if txt.endswith(txt[-L:][:L]) and txt[-2*L:-L] == txt[-L:]:
            rep = L
    print("PROMPT %-40s -> %r" % (p[:40], txt[:300]))
    print("       len=%d 尾部最长重复块=%d" % (len(txt), rep))
PY
echo "=== 停服还卡 ==="
[ -f "$PIDF" ] && kill -TERM -"$(cat $PIDF)" 2>/dev/null
sleep 15; docker rm -f dsv41-ct-int4 >/dev/null 2>&1
rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{printf "%.2f ", $NF/1073741824} END{print "GiB"}'
echo "FIXED_ARM_DONE $(date +%T)"
