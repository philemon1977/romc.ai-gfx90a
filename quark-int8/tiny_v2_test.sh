#!/usr/bin/env bash
# Mechanics + equivalence test of expert-cache V2 on the tiny INT8 checkpoint.
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
IMG=vllm/vllm-openai-rocm:nightly
PORT=8125
run_server () {   # $1 = name, $2.. = extra env
  docker rm -f "$1" >/dev/null 2>&1 || true
  shift_args=("${@:2}")
  docker run -d --name "$1" --network host --device /dev/kfd --device /dev/dri \
    --group-add video --shm-size 8G -e HIP_VISIBLE_DEVICES=0 -e VLLM_ENGINE_READY_TIMEOUT_S=600 \
    "${shift_args[@]}" -v "$REPO:/work" "$IMG" \
    /work/tiny_int8 --served-model-name tiny --port $PORT --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.06 --max-model-len 1024 --max-num-seqs 4 \
    --enforce-eager --trust-remote-code >/dev/null
  for i in $(seq 1 60); do
    curl -s -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "$1 healthy after $((i*5))s"; return 0; }
    sleep 5
  done
  echo "$1 FAILED to become healthy"; docker logs "$1" 2>&1 | tail -5 | cut -c1-160; return 1
}

echo "=== 1) baseline: cache OFF ==="
run_server tiny-nocache || exit 1
docker run --rm --entrypoint bash --network host -v "$REPO:/work" "$IMG" \
  -c "python3 /work/tiny_probe.py $PORT nocache" 2>&1 | tail -2
docker rm -f tiny-nocache >/dev/null 2>&1; sleep 8

echo "=== 2) cache ON (V2, 25% cold, pool=2) ==="
run_server tiny-v2 \
  -e EXPERT_CACHE=1 -e EXPERT_CACHE_MODE=slot -e EXPERT_CACHE_PCT=50 \
  -e EXPERT_CACHE_POOL=2 -e EXPERT_CACHE_HOTLIST=/ec/hotlist.npz \
  -e EXPERT_CACHE_STATS=/work/tiny_v2_stats.json -e PYTHONPATH=/ec \
  -v "$REPO/expert_cache:/ec" || exit 1
docker logs tiny-v2 2>&1 | grep -E "expert-cache-v2" | head -3 | cut -c1-150
docker run --rm --entrypoint bash --network host -v "$REPO:/work" "$IMG" \
  -c "python3 /work/tiny_probe.py $PORT cached" 2>&1 | tail -2
echo "=== 3) cache stats ==="
docker exec tiny-v2 bash -c "python3 -c \"
import json;d=json.load(open('/work/tiny_v2_stats.json'));print(json.dumps(d['global'],indent=1)) ; print('layer sample:', list(d['sample_layers'].items())[:2])\"" 2>/dev/null | head -25
echo "=== 4) equivalence ==="
python3 - <<'PY'
import json
a=json.load(open('/home/qiba/ROCm.AI/quark-int8/tiny_nocache.json'))
b=json.load(open('/home/qiba/ROCm.AI/quark-int8/tiny_cached.json'))
print("nocache:", a['mean_nll'], a['sha'])
print("cached :", b['mean_nll'], b['sha'])
print("EXACT MATCH" if a['sha']==b['sha'] and abs(a['mean_nll']-b['mean_nll'])<1e-9 else "MISMATCH")
PY
