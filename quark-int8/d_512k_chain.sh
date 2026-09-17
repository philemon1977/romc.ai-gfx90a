#!/usr/bin/env bash
# Chain: wait for the isolation run -> launch the 512K candidate (YaRN 2.0x, seqs 8)
# -> measure decode speed -> needle tests at ~300K and ~500K. All output to logs/d_512k.log
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8; IMG=vllm/vllm-openai-rocm:nightly
LOG="$REPO/logs/d_512k.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== chain start $(date -u +%FT%TZ) ==="
# 1) wait for isolation run to finish
for i in $(seq 1 120); do
  st=$(docker ps -a --filter name=ornith-iso --format "{{.Status}}")
  case "$st" in Exited*|"") echo "iso finished: $st"; break;; esac
  sleep 30
done
echo "--- isolation result ---"; tail -6 "$REPO/logs/speed_iso.log"
docker rm -f ornith-iso >/dev/null 2>&1 || true
sleep 20
# 2) launch the 512K candidate
NAME=ornith-d512k
OVR='{"max_position_embeddings":524288,"text_config":{"max_position_embeddings":524288,"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144,"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}}}'
for i in $(seq 1 60); do
  f=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "${f:-0}" -ge 62 ] && break; sleep 10
done
docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME --network host --device /dev/kfd --device /dev/dri --group-add video \
  --shm-size 64G --ulimit memlock=-1:-1 -e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 $IMG \
  /mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-Quark-Int8-Attn \
  --served-model-name Ornith-d512k --port 8100 --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.975 --max-model-len 520000 --max-num-batched-tokens 2048 \
  --max-num-seqs 8 --hf-overrides "$OVR" \
  --language-model-only --trust-remote-code --moe-backend triton \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' >/dev/null
"$REPO/watch_container.sh" $NAME 900 8100 || exit 1
docker logs $NAME 2>&1 | grep -oE "GPU KV cache size: [0-9,]+ tokens.*" | tail -1
# 3) speed + needle
docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG -c "python3 -u - <<'PY'
import json,time,urllib.request
body=json.dumps({'model':'Ornith-d512k','prompt':'Count slowly:','max_tokens':128,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8100/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=900)); dt=time.time()-t0
print(f'[d512k] short-context decode {d[\"usage\"][\"completion_tokens\"]} tok in {dt:.1f}s -> {d[\"usage\"][\"completion_tokens\"]/dt:.2f} tok/s')
PY"
for T in 262144 438000; do
  echo "=== needle @ target $T (~$((T*114/100)) actual) $(date -u +%H:%M:%S) ==="
  docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG \
    -c "python3 -u /work/longctx_test.py 8100 Ornith-d512k $T 512" 2>&1 | tail -3
done
echo "=== chain done $(date -u +%FT%TZ) ==="
