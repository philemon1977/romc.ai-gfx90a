#!/usr/bin/env bash
# Real-model INT8 verification: wait for the TP8 server, run chat/completion
# requests, measure decode throughput, and store the evidence.
set -uo pipefail
PORT=${PORT:-8100}
OUT=${OUT:-/home/qiba/ROCm.AI/quark-int8/EVIDENCE_real_int8.txt}
MODEL=${MODEL:-Ornith-1.5-397B-Int8}

exec > >(tee "$OUT") 2>&1
echo "=== Ornith-1.5-397B Quark INT8 / vLLM TP8 on 8x MI250X (gfx90a) ==="
date -u +"started %FT%TZ"

echo "--- waiting for /health ---"
for i in $(seq 1 120); do
  if curl -s -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "health OK after $((i*10))s"; break
  fi
  sleep 10
done
curl -s -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || { echo "SERVER NOT HEALTHY"; exit 1; }

echo "--- model list ---"
curl -s -m 10 "http://127.0.0.1:${PORT}/v1/models" | head -c 400; echo

echo "--- chat completion (greedy) ---"
curl -s -m 180 "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句中文说明 AMD Instinct MI250X 的显存容量和架构代号。\"}],\"max_tokens\":64,\"temperature\":0}" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content']); print('usage:', d['usage'])" 2>/dev/null || echo "chat request failed"

echo "--- completion throughput (256 tokens) ---"
python3 - "$PORT" "$MODEL" <<'PY'
import json, sys, time, urllib.request
port, model = sys.argv[1], sys.argv[2]
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=json.dumps({"model": model, "prompt": "Explain in detail how INT8 weight-only quantization reduces memory bandwidth pressure during LLM decoding.",
                     "max_tokens": 256, "temperature": 0}).encode(),
    headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=600) as r:
    d = json.load(r)
dt = time.time() - t0
tok = d["usage"]["completion_tokens"]
print(f"generated {tok} tokens in {dt:.2f}s -> {tok/dt:.2f} tok/s (single request)")
print("sample:", d["choices"][0]["text"][:200].replace("\n", " "))
PY

echo "--- GPU memory after serving ---"
rocm-smi --showmeminfo vram --csv | tail -n +2 | awk -F, '{printf "%s used %.1f/%.1f GB\n", $1, $3/1e9, $2/1e9}'
echo "=== done ==="
