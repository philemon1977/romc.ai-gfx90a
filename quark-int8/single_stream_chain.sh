#!/usr/bin/env bash
# Single-stream TPS focus with CT-Int4 (text-only architecture to bypass the
# vision-tower loading bug).
#   S1: text-only int4 @512K + ngram  -> load check, single-stream TPS, NLL
#   S2: text-only int4 @32K  + ngram  -> isolates the max-model-len penalty
#   S3: text-only int4 @512K + MTP    -> step-reduction comparison for single stream
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8; IMG=vllm/vllm-openai-rocm:nightly
LOG="$REPO/logs/single_stream.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
MDL=/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-CT-Int4-W4A16
NGRAM='{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_min":5,"prompt_lookup_max":5}'
MTP='{"method":"mtp","num_speculative_tokens":1}'
QCOVR='{"quant_method":"compressed-tensors","format":"pack-quantized","config_groups":{"group_0":{"targets":["Linear"],"input_activations":null,"weights":{"num_bits":4,"type":"int","symmetric":true,"strategy":"group","group_size":128,"dynamic":false,"actorder":null}}},"ignore":["lm_head","model.visual.*","visual.*","re:.*visual.*","mtp.*","*mlp.gate","*mlp.gate.linear","*shared_expert_gate*","*.shared_expert.*","*.linear_attn.*","*.self_attn.*","re:.*embed_tokens.*","re:.*norm.*","re:.*conv.*"],"quantization_status":"compressed"}'
ROPE2='{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144,"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}'
ROPE0='{"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}'

wait_gpus () {
  for _ in $(seq 1 240); do
    busy=$(docker ps --format '{{.Names}}' | grep -c '^ornith-' || true)
    free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    [ "$busy" = "0" ] && [ "${free:-0}" -ge 62 ] && return 0
    sleep 15
  done
  return 1
}

run () {  # tag maxlen spec maxseqs rope
  local tag=$1 maxlen=$2 spec=$3 maxseqs=$4 rope=$5
  local name=ornith-$tag
  docker rm -f "$name" >/dev/null 2>&1 || true
  wait_gpus || { echo "GPUs busy"; return 1; }
  local ovr; ovr=$(python3 - "$maxlen" "$rope" "$QCOVR" <<'PY'
import json, sys
maxlen = int(sys.argv[1]); rope = json.loads(sys.argv[2]); qc = json.loads(sys.argv[3])
print(json.dumps({
    "architectures": ["Qwen3_5MoeForCausalLM"],      # text-only: no vision tower at all
    "model_type": "qwen3_5_moe",
    "quantization_config": qc,
    "text_config": {"max_position_embeddings": maxlen, "rope_parameters": rope},
}))
PY
)
  echo "=== $tag: maxlen=$maxlen spec=$(echo $spec | cut -c1-40) seqs=$maxseqs $(date -u +%H:%M:%S) ==="
  docker run -d --name "$name" --network host --device /dev/kfd --device /dev/dri --group-add video \
    --shm-size 64G --ulimit memlock=-1:-1 -e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 $IMG \
    $MDL --served-model-name "Ornith-$tag" --port 8100 --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.975 --max-model-len "$maxlen" --max-num-batched-tokens 4096 \
    --max-num-seqs "$maxseqs" --hf-overrides "$ovr" --trust-remote-code \
    --moe-backend triton --speculative-config "$spec" >/dev/null
  if ! "$REPO/watch_container.sh" "$name" 1200 8100; then echo "$tag FAILED"; return 1; fi
  docker logs "$name" 2>&1 | grep -oE "Using [A-Za-z_]+ MoE backend[^.]*\.|Resolved architecture: [A-Za-z0-9_]+|GPU KV cache size: [0-9,]+ tokens.*" | tail -3
  # single-stream TPS: one request, 256 output tokens
  docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG -c "python3 -u - <<PY
import json,time,urllib.request
body=json.dumps({'model':'Ornith-$tag','prompt':'Explain in detail why int4 quantization reduces memory bandwidth pressure during decoding.','max_tokens':256,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8100/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=1800)); dt=time.time()-t0
u=d['usage']; print('[$tag] SINGLE-STREAM: %d tok in %.1fs -> %.2f tok/s' % (u['completion_tokens'],dt,u['completion_tokens']/dt))
PY"
  curl -s localhost:8100/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{" | sed 's/^/  /'
  docker rm -f "$name" >/dev/null 2>&1 || true; sleep 15
}

run S1-int4-txt-512k-ngram 520000 "$NGRAM" 16 "$ROPE2"
run S2-int4-txt-32k-ngram   32768  "$NGRAM" 16 "$ROPE0"
run S3-int4-txt-512k-mtp    520000 "$MTP"   16 "$ROPE2"
echo "=== single_stream done $(date -u +%FT%TZ) ==="
