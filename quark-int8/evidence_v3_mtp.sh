#!/usr/bin/env bash
# Evidence collection for the MXFP4 (W4A16) checkpoint served with MTP speculative decoding.
# Usage: PORT=8100 MODEL=Ornith-1.5-397B-MXFP4-W4A16 OUT=... ./evidence_v3_mtp.sh
set -uo pipefail
PORT=${PORT:-8100}
MODEL=${MODEL:-Ornith-1.5-397B-MXFP4-W4A16}
OUT=${OUT:-/home/qiba/ROCm.AI/quark-int8/EVIDENCE_v3_mxfp4_mtp.txt}
CONTAINER=${CONTAINER:-ornith-mxfp4-mtp}

exec > >(tee "$OUT") 2>&1
echo "=== Ornith-1.5-397B MXFP4 (W4A16) + MTP speculative decoding / TP8 on 8x MI250X ==="
date -u +"started %FT%TZ"

echo "--- engine facts ---"
docker logs "$CONTAINER" 2>&1 | grep -iE "quantization=|Using .*Mxfp4.* backend|quant_method|Resolved architecture|speculative|GPU KV cache size|Loading weights took|init engine" | tail -10 | cut -c1-200

echo "--- health ---"
for i in $(seq 1 30); do curl -s -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo "health OK"; break; }; sleep 10; done

echo "--- chat completion (greedy, 512 tokens) ---"
curl -s -m 600 "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"用中文分三点说明 INT4 量化在 MoE 大模型推理中的收益与代价。\"}],\"max_tokens\":512,\"temperature\":0}" \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['choices'][0]['message']['content'][-800:]);print('usage:',d['usage'])" 2>/dev/null || echo "chat failed"

echo "--- decode throughput (256 tokens, greedy) ---"
python3 - "$PORT" "$MODEL" <<'PY'
import json, sys, time, urllib.request
port, model = sys.argv[1], sys.argv[2]
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
    data=json.dumps({"model": model, "prompt": "Explain how 4-bit weight quantization lowers memory bandwidth pressure in mixture-of-experts decoding.",
                     "max_tokens": 256, "temperature": 0}).encode(),
    headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=900) as r:
    d = json.load(r)
dt = time.time() - t0
tok = d["usage"]["completion_tokens"]
print(f"generated {tok} tokens in {dt:.2f}s -> {tok/dt:.2f} tok/s (single request, MTP on)")
print("sample:", d["choices"][0]["text"][:160].replace("\n", " "))
PY

echo "--- speculative decoding metrics ---"
curl -s -m 10 "http://127.0.0.1:${PORT}/metrics" | grep -E "^vllm:spec_decode_(num_drafts|num_draft_tokens|num_accepted_tokens)_total" | awk '{print $1, $2}'

echo "--- quality (mean token NLL) ---"
docker run --rm --entrypoint bash --network host -v /home/qiba/ROCm.AI/quark-int8:/work vllm/vllm-openai-rocm:nightly \
  -c "python3 /work/nll_probe.py ${PORT} ${MODEL} v3-mxfp4-mtp 2>&1 | tail -1"

echo "--- GPU memory ---"
rocm-smi --showmeminfo vram --csv | tail -n +2 | awk -F, '{printf "%s used %.1f/%.1f GB\n", $1, $3/1e9, $2/1e9}'
echo "=== done ==="
