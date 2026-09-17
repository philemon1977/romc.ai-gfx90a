#!/usr/bin/env bash
# T2 (int8 @256K + ngram) is already loading -> wait for it, measure, release GPUs.
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8; IMG=vllm/vllm-openai-rocm:nightly
exec > >(tee -a "$REPO/logs/tps256b.log") 2>&1
echo "=== T2 finish $(date -u +%H:%M:%S) ==="
"$REPO/watch_container.sh" ornith-T2-int8-256k-ngram5 900 8100 || { docker rm -f ornith-T2-int8-256k-ngram5 >/dev/null 2>&1; exit 1; }
docker logs ornith-T2-int8-256k-ngram5 2>&1 | grep -oE "GPU KV cache size: [0-9,]+ tokens.*|Available KV cache memory: [0-9.]+ GiB" | tail -2
docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG -c "python3 -u - <<'PY'
import json,time,urllib.request
body=json.dumps({'model':'Ornith-T2-int8-256k-ngram5','prompt':'Explain in detail why int4 quantization reduces memory bandwidth pressure during decoding.','max_tokens':256,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8100/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=1800)); dt=time.time()-t0
u=d['usage']; print('[T2-int8-256k-ngram5] SINGLE-STREAM %.2f tok/s (%d tok in %.1fs)' % (u['completion_tokens']/dt,u['completion_tokens'],dt))
PY"
curl -s localhost:8100/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{" | sed 's/^/  /'
docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG -c "python3 -u /work/nll_probe.py 8100 Ornith-T2-int8-256k-ngram5 int8-256k" 2>&1 | tail -1
docker rm -f ornith-T2-int8-256k-ngram5 >/dev/null 2>&1
echo "=== T2 done, GPUs released $(date -u +%H:%M:%S) ==="
