#!/usr/bin/env bash
# A/B experiments for the "long max-model-len costs 2.6x decode" problem.
#  A) maxlen 32768, seqs 16, util 0.96, ngram   -> isolates maxlen from ngram/seqs/util
#  B) maxlen 520000, seqs 8, util 0.975, ngram, VLLM_KV_CACHE_LAYOUT=NHD -> layout lever
# Each run: launch -> wait healthy -> measure short-context decode tok/s.
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8; IMG=vllm/vllm-openai-rocm:nightly
LOG="$REPO/logs/attn_ab.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
MDL=/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-Quark-Int8-Attn
NGRAM='{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_min":5,"prompt_lookup_max":5}'
YARN='{"max_position_embeddings":524288,"text_config":{"max_position_embeddings":524288,"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144,"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}}}'

run_case () {  # $1=tag $2=maxlen $3=seqs $4=util $5=extra env pairs (space separated) ; $6 = 1 if YaRN
  local tag=$1 maxlen=$2 seqs=$3 util=$4 extra=$5 yarn=$6
  local name=ornith-$tag
  docker rm -f "$name" >/dev/null 2>&1 || true
  local need=$(python3 -c "print(int(${util}*63.98)+1)")
  for i in $(seq 1 90); do
    f=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    [ "${f:-0}" -ge "$need" ] && break; sleep 10
  done
  local yarnarg=()
  [ "$yarn" = "1" ] && yarnarg=(--hf-overrides "$YARN")
  echo "=== case $tag: maxlen=$maxlen seqs=$seqs util=$util extra='$extra' yarn=$yarn $(date -u +%H:%M:%S) ==="
  docker run -d --name "$name" --network host --device /dev/kfd --device /dev/dri --group-add video \
    --shm-size 64G --ulimit memlock=-1:-1 -e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 $extra -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 $IMG \
    $MDL --served-model-name "Ornith-$tag" --port 8100 --tensor-parallel-size 8 \
    --gpu-memory-utilization "$util" --max-model-len "$maxlen" --max-num-batched-tokens 2048 \
    --max-num-seqs "$seqs" "${yarnarg[@]}" --language-model-only --trust-remote-code \
    --moe-backend triton --speculative-config "$NGRAM" >/dev/null
  "$REPO/watch_container.sh" "$name" 900 8100 || { echo "case $tag FAILED"; return 1; }
  docker logs "$name" 2>&1 | grep -oE "kv cache group sizes \[[0-9, ]+\]|kv lcm block sizes [0-9]+|GPU KV cache size: [0-9,]+ tokens.*|Using .*KV cache layout[^\"]*" | tail -3
  docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG -c "python3 -u - <<PY
import json,time,urllib.request
body=json.dumps({'model':'Ornith-$tag','prompt':'Count slowly:','max_tokens':128,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8100/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=900)); dt=time.time()-t0
print('[$tag] short-ctx decode %.2f tok/s' % (d['usage']['completion_tokens']/dt))
PY"
  docker rm -f "$name" >/dev/null 2>&1 || true
  sleep 15
}

run_case A32k-ngram 32768 16 0.96 "" 0
run_case B512k-nhd 520000 8 0.975 "-e VLLM_KV_CACHE_LAYOUT=NHD" 1
echo "=== attn_ab done $(date -u +%FT%TZ) ==="
