#!/usr/bin/env bash
# Finish the already-loading D1 (INT8 512K seqs8) case: wait healthy, sweep 1/4/8,
# then release the GPUs so the int4 chain (waiting in wait_gpus) can start.
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8; IMG=vllm/vllm-openai-rocm:nightly
LOG="$REPO/logs/conc_chain2.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== D1 finish $(date -u +%H:%M:%S) ==="
"$REPO/watch_container.sh" ornith-D1-locked512k 900 8100 || { docker rm -f ornith-D1-locked512k >/dev/null 2>&1; exit 1; }
docker logs ornith-D1-locked512k 2>&1 | grep -oE "GPU KV cache size: [0-9,]+ tokens.*" | tail -1
docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG \
  -c "python3 -u /work/bench_concurrency.py 8100 Ornith-D1-locked512k 1,4,8 400 128" 2>&1 | tail -5
docker rm -f ornith-D1-locked512k >/dev/null 2>&1
echo "=== D1 done, GPUs released $(date -u +%H:%M:%S) ==="
