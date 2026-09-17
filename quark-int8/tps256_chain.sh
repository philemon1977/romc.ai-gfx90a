#!/usr/bin/env bash
# 256K context (native window, no YaRN) + single-stream TPS focus.
#   T1: CT-Int4  text-only @256K + ngram(5)  -> candidate: TPS, NLL, sweep 1/4/8/16
#   T2: Quark-INT8 text-only @256K + ngram(5)-> does int4 actually help single-stream TPS?
#   T3: CT-Int4  text-only @256K + MTP(1)    -> spec method comparison for single stream
#   T4: CT-Int4  text-only @256K + ngram(10) -> deeper drafts (more tokens per step)
#   T5: CT-Int4  text-only @32K  + ngram(5)  -> isolates the max-model-len penalty
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8; IMG=vllm/vllm-openai-rocm:nightly
LOG="$REPO/logs/tps256.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
MDL_INT4=/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-CT-Int4-W4A16
MDL_INT8=/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-Quark-Int8-Attn
QCOVR='{"quant_method":"compressed-tensors","format":"pack-quantized","config_groups":{"group_0":{"targets":["Linear"],"input_activations":null,"weights":{"num_bits":4,"type":"int","symmetric":true,"strategy":"group","group_size":128,"dynamic":false,"actorder":null}}},"ignore":["lm_head","model.visual.*","visual.*","re:.*visual.*","mtp.*","*mlp.gate","*mlp.gate.linear","*shared_expert_gate*","*.shared_expert.*","*.linear_attn.*","*.self_attn.*","re:.*embed_tokens.*","re:.*norm.*","re:.*conv.*"],"quantization_status":"compressed"}'

wait_gpus () {
  for _ in $(seq 1 240); do
    busy=$(docker ps --format '{{.Names}}' | grep -c '^ornith-' || true)
    free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    [ "$busy" = "0" ] && [ "${free:-0}" -ge 62 ] && return 0
    sleep 15
  done
  return 1
}

run () {  # tag mdl maxlen spec_text seqs do_sweep
  local tag=$1 mdl=$2 maxlen=$3 spec=$4 seqs=$5 sweep=$6
  local name=ornith-$tag
  docker rm -f "$name" >/dev/null 2>&1 || true
  wait_gpus || { echo "GPUs busy"; return 1; }
  local ovr=""
  if [ "$mdl" = "$MDL_INT4" ]; then
    ovr=$(python3 - "$QCOVR" <<'PY'
import json, sys
print(json.dumps({"architectures": ["Qwen3_5MoeForCausalLM"], "model_type": "qwen3_5_moe",
                  "quantization_config": json.loads(sys.argv[1])}))
PY
)
  fi
  local ovrarg=(); [ -n "$ovr" ] && ovrarg=(--hf-overrides "$ovr")
  echo "=== $tag maxlen=$maxlen seqs=$seqs spec=$(echo "$spec" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["method"], d["num_speculative_tokens"])') $(date -u +%H:%M:%S) ==="
  docker run -d --name "$name" --network host --device /dev/kfd --device /dev/dri --group-add video \
    --shm-size 64G --ulimit memlock=-1:-1 -e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 $IMG \
    $mdl --served-model-name "Ornith-$tag" --port 8100 --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.975 --max-model-len "$maxlen" --max-num-batched-tokens 4096 \
    --max-num-seqs "$seqs" "${ovrarg[@]}" --trust-remote-code \
    --moe-backend triton --speculative-config "$spec" >/dev/null
  if ! "$REPO/watch_container.sh" "$name" 1200 8100; then echo "$tag FAILED"; return 1; fi
  docker logs "$name" 2>&1 | grep -oE "Using [A-Za-z_]+ MoE backend[^.]*\.|Resolved architecture: [A-Za-z0-9_]+|GPU KV cache size: [0-9,]+ tokens.*" | tail -3
  docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG -c "python3 -u - <<PY
import json,time,urllib.request
body=json.dumps({'model':'Ornith-$tag','prompt':'Explain in detail why int4 quantization reduces memory bandwidth pressure during decoding.','max_tokens':256,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8100/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=1800)); dt=time.time()-t0
u=d['usage']; print('[$tag] SINGLE-STREAM %.2f tok/s (%d tok in %.1fs)' % (u['completion_tokens']/dt,u['completion_tokens'],dt))
PY"
  curl -s localhost:8100/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{" | sed 's/^/  /'
  if [ "$sweep" = "1" ]; then
    docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG \
      -c "python3 -u /work/bench_concurrency.py 8100 Ornith-$tag 1,4,8,16 400 128" 2>&1 | tail -5
  fi
  docker rm -f "$name" >/dev/null 2>&1 || true; sleep 15
}

N5='{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_min":5,"prompt_lookup_max":5}'
N10='{"method":"ngram","num_speculative_tokens":10,"prompt_lookup_min":5,"prompt_lookup_max":5}'
M1='{"method":"mtp","num_speculative_tokens":1}'
run T1-int4-256k-ngram5  "$MDL_INT4" 262144 "$N5"  16 1
run T2-int8-256k-ngram5  "$MDL_INT8" 262144 "$N5"  16 0
run T3-int4-256k-mtp     "$MDL_INT4" 262144 "$M1"  16 0
run T4-int4-256k-ngram10 "$MDL_INT4" 262144 "$N10" 16 0
run T5-int4-32k-ngram5   "$MDL_INT4" 32768  "$N5"  16 0
echo "=== tps256 done $(date -u +%FT%TZ) ==="
