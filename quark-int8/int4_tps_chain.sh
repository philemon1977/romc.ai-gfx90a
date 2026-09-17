#!/usr/bin/env bash
# CT-Int4 single-stream TPS at 256K. Uses the ORIGINAL architecture (the checkpoint
# contains model.visual.* tensors, so a text-only arch fails) + the ignore list
# rewritten as REGEX (vLLM matches compressed-tensors ignore patterns as regex).
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8; IMG=vllm/vllm-openai-rocm:nightly
LOG="$REPO/logs/int4_tps.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
MDL=/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-CT-Int4-W4A16
QCOVR='{"quantization_config":{"quant_method":"compressed-tensors","format":"pack-quantized","config_groups":{"group_0":{"targets":["Linear"],"input_activations":null,"weights":{"num_bits":4,"type":"int","symmetric":true,"strategy":"group","group_size":128,"dynamic":false,"actorder":null}}},"ignore":["re:^lm_head","re:.*visual\\..*","re:^mtp\\..*","re:.*mlp\\.gate$","re:.*mlp\\.gate\\.linear$","re:.*shared_expert_gate.*","re:.*\\.shared_expert\\..*","re:.*\\.linear_attn\\..*","re:.*\\.self_attn\\..*","re:.*embed_tokens.*","re:.*norm.*","re:.*conv.*"],"quantization_status":"compressed"}}'

wait_gpus () {
  for _ in $(seq 1 240); do
    busy=$(docker ps --format '{{.Names}}' | grep -c '^ornith-' || true)
    free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    [ "$busy" = "0" ] && [ "${free:-0}" -ge 62 ] && return 0
    sleep 15
  done
  return 1
}

run () {  # tag maxlen spec seqs sweep
  local tag=$1 maxlen=$2 spec=$3 seqs=$4 sweep=$5
  local name=ornith-$tag
  docker rm -f "$name" >/dev/null 2>&1 || true
  wait_gpus || { echo "GPUs busy"; return 1; }
  echo "=== $tag maxlen=$maxlen seqs=$seqs $(date -u +%H:%M:%S) ==="
  docker run -d --name "$name" --network host --device /dev/kfd --device /dev/dri --group-add video \
    --shm-size 64G --ulimit memlock=-1:-1 -e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 $IMG \
    $MDL --served-model-name "Ornith-$tag" --port 8100 --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.975 --max-model-len "$maxlen" --max-num-batched-tokens 2048 \
    --max-num-seqs "$seqs" --hf-overrides "$QCOVR" --language-model-only --trust-remote-code \
    --moe-backend triton --speculative-config "$spec" >/dev/null
  if ! "$REPO/watch_container.sh" "$name" 1200 8100; then echo "$tag FAILED"; return 1; fi
  docker logs "$name" 2>&1 | grep -oE "Using [A-Za-z_]+ MoE backend[^.]*\.|GPU KV cache size: [0-9,]+ tokens.*" | tail -2
  docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG -c "python3 -u - <<PY
import json,time,urllib.request
body=json.dumps({'model':'Ornith-$tag','prompt':'Explain in detail why int4 quantization reduces memory bandwidth pressure during decoding.','max_tokens':256,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8100/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=1800)); dt=time.time()-t0
u=d['usage']; print('[$tag] SINGLE-STREAM %.2f tok/s (%d tok in %.1fs)' % (u['completion_tokens']/dt,u['completion_tokens'],dt))
PY"
  curl -s localhost:8100/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{" | sed 's/^/  /'
  docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG -c "python3 -u /work/nll_probe.py 8100 Ornith-$tag $tag" 2>&1 | tail -1
  if [ "$sweep" = "1" ]; then
    docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG \
      -c "python3 -u /work/bench_concurrency.py 8100 Ornith-$tag 1,4,8,16 400 128" 2>&1 | tail -5
  fi
  docker rm -f "$name" >/dev/null 2>&1 || true; sleep 15
}

N5='{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_min":5,"prompt_lookup_max":5}'
N10='{"method":"ngram","num_speculative_tokens":10,"prompt_lookup_min":5,"prompt_lookup_max":5}'
M1='{"method":"mtp","num_speculative_tokens":1}'
run U1-int4-256k-ngram5  262144 "$N5"  16 1
run U2-int4-256k-ngram10 262144 "$N10" 16 0
run U3-int4-256k-mtp     262144 "$M1"  16 0
run U4-int4-32k-ngram5   32768  "$N5"  16 0
echo "=== int4_tps done $(date -u +%FT%TZ) ==="
