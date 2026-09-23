#!/bin/bash
# 快速验证：gfx90a 稀疏解码移植是否生效（起服 → 就绪 → 事实召回 + 冒烟 → 停服）
set -u
R=/home/qiba/ROCm.AI/quark-int8
LOG=$R/logs/glm53_quick_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== 快速验证 $(date +%T) ==="
PORT=8121 MAX_MODEL_LEN=1048576 MAX_NUM_SEQS=2 MAX_NUM_BATCHED_TOKENS=2048 ENFORCE_EAGER=1 \
  bash /home/qiba/ai/models/ZhipuAI/launcher/glm53_int4w4a16_vllm_rocmnightly0918_32k_8121_mi250dx8.sh || exit 1
L=$(ls -t /home/qiba/ai/logs/glm53/server-8121-*.log | head -1); echo "SERVER_LOG=$L"
for i in $(seq 1 120); do
  grep -qa "Application startup complete" "$L" 2>/dev/null && { echo "READY $(date +%T)"; break; }
  grep -qaE "AttributeError|RuntimeError|assert|Error" "$L" 2>/dev/null && { echo "FAILED $(date +%T)"; grep -aE "AttributeError|RuntimeError|assert|Error" "$L" | grep -avE "warn|WARNING" | tail -5 | cut -c1-200; break; }
  docker ps -q -f name=glm53-int4 | grep -q . || { echo "CONTAINER GONE"; tail -20 "$L" | cut -c1-200; break; }
  sleep 20
done
if grep -qa "Application startup complete" "$L" 2>/dev/null; then
  echo "== 取证 =="
  grep -aE "TritonW4A16LinearKernel|WNA16 MoE backend|kernel module = mi250|Model loading took|GPU KV cache size|Available KV|maximum concurrency" "$L" | tail -7 | cut -c1-180
  echo "== 事实召回 =="; python3 $R/fact_recall_probe.py 8121 glm-5.3
  echo "== 算术冒烟 =="
  python3 - <<PY
import json, urllib.request
for p in ["What is 17*19? Answer: 17*19 =", "用一句话说明水的化学式。"]:
    d = json.dumps({"model": "glm-5.3", "prompt": p, "max_tokens": 32, "temperature": 0}).encode()
    r = urllib.request.Request("http://127.0.0.1:8121/v1/completions", d, {"Content-Type": "application/json"})
    print(" ", repr(p[:24]), "->", repr(json.load(urllib.request.urlopen(r, timeout=900))["choices"][0]["text"][:120]))
PY
fi
echo "== 停服 $(date +%T) =="; docker rm -f glm53-int4 >/dev/null 2>&1; sleep 8
rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{printf "%.1f ", $NF/1073741824} END{print "GiB"}'
echo "QUICK_DONE $(date +%T)"
