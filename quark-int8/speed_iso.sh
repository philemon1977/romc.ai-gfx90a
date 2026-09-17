#!/usr/bin/env bash
# Isolation run: same as astep (maxlen 262144, seqs 2, util 0.975, MTP, no YaRN)
# then measure short-context decode speed. Log -> logs/speed_iso.log
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8; IMG=vllm/vllm-openai-rocm:nightly
NAME=ornith-iso; LOG="$REPO/logs/speed_iso.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
docker rm -f $NAME >/dev/null 2>&1 || true
for i in $(seq 1 60); do
  f=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "${f:-0}" -ge 62 ] && break; sleep 10
done
docker run -d --name $NAME --network host --device /dev/kfd --device /dev/dri --group-add video \
  --shm-size 64G --ulimit memlock=-1:-1 -e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 $IMG \
  /mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-Quark-Int8-Attn \
  --served-model-name Ornith-iso --port 8100 --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.975 --max-model-len 262144 --max-num-batched-tokens 2048 \
  --max-num-seqs 2 --language-model-only --trust-remote-code --moe-backend triton \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' >/dev/null
"$REPO/watch_container.sh" $NAME 900 8100 || exit 1
docker logs $NAME 2>&1 | grep -oE "GPU KV cache size: [0-9,]+ tokens.*" | tail -1
docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG -c "python3 -u - <<'PY'
import json,time,urllib.request
for n in (48, 128):
    body=json.dumps({'model':'Ornith-iso','prompt':'Count slowly:','max_tokens':n,'temperature':0,'ignore_eos':True}).encode()
    req=urllib.request.Request('http://127.0.0.1:8100/v1/completions',data=body,headers={'Content-Type':'application/json'})
    t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=900)); dt=time.time()-t0
    print(f'decode {d[\"usage\"][\"completion_tokens\"]} tok in {dt:.1f}s -> {d[\"usage\"][\"completion_tokens\"]/dt:.2f} tok/s')
PY"
echo "=== iso done ==="
