#!/usr/bin/env bash
# SINGLE serialized orchestrator (avoids the two-waiter race that killed U3).
# Queue: U3 int4+MTP(TP8) -> U4 int4+32K(TP8) -> PP8 int4+MTP(TP1xPP8)
# Each case: strict wait (no ornith container running/leftover + free>=MIN_FREE_GIB
# on EVERY die + settle) -> launch -> wait healthy -> measure -> report -> remove.
# MIN_FREE_GIB=62 (default, exclusive) or e.g. 45 to coexist with another session
# (measurements then are lower bounds; record that in the log).
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8; IMG=vllm/vllm-openai-rocm:nightly
LOG="$REPO/logs/queue.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
MDL=/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-CT-Int4-W4A16
MIN_FREE_GIB="${MIN_FREE_GIB:-62}"
QCOVR='{"quantization_config":{"quant_method":"compressed-tensors","format":"pack-quantized","config_groups":{"group_0":{"targets":["Linear"],"input_activations":null,"weights":{"num_bits":4,"type":"int","symmetric":true,"strategy":"group","group_size":128,"dynamic":false,"actorder":null}}},"ignore":["re:^lm_head","re:.*visual\\..*","re:^mtp\\..*","re:.*mlp\\.gate$","re:.*mlp\\.gate\\.linear$","re:.*shared_expert_gate.*","re:.*\\.shared_expert\\..*","re:.*\\.linear_attn\\..*","re:.*\\.self_attn\\..*","re:.*embed_tokens.*","re:.*norm.*","re:.*conv.*"],"quantization_status":"compressed"}}'

strict_wait () {
  local ok=0
  for _ in $(seq 1 240); do
    local alive free
    alive=$(docker ps -a --format '{{.Names}}|{{.Status}}' | grep '^ornith-' | grep -c 'Up' || true)
    free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 \
           | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    if [ "$alive" = "0" ] && [ "${free:-0}" -ge "$MIN_FREE_GIB" ]; then
      ok=$((ok+1)); [ "$ok" -ge 2 ] && { sleep 20; return 0; }
    else ok=0; fi
    sleep 20
  done
  return 1
}

run () {  # tag maxlen seqs tp pp spec sweep
  local tag=$1 maxlen=$2 seqs=$3 tp=$4 pp=$5 spec=$6 sweep=$7
  local name=ornith-$tag
  docker rm -f "$name" >/dev/null 2>&1 || true
  # clean any other leftover ornith containers of *this* queue
  for c in $(docker ps -a --format '{{.Names}}' | grep '^ornith-' || true); do docker rm -f "$c" >/dev/null 2>&1; done
  strict_wait || { echo "$tag: 等不到 ${MIN_FREE_GIB} GiB 空闲，跳过"; return 1; }
  echo "=== $tag maxlen=$maxlen seqs=$seqs TP=$tp PP=$pp spec=${spec:-none} $(date -u +%H:%M:%S) ==="
  docker run -d --name "$name" --network host --device /dev/kfd --device /dev/dri --group-add video \
    --shm-size 64G --ulimit memlock=-1:-1 -e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    -e TRITON_CACHE_DIR=/triton_cache -e TRITON_KERNEL_CACHE_DIR=/triton_cache \
    -v "$REPO/triton_cache:/triton_cache" \
    -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 $IMG \
    $MDL --served-model-name "Ornith-$tag" --port 8100 --tensor-parallel-size "$tp" \
    --pipeline-parallel-size "$pp" --gpu-memory-utilization "${UTIL:-0.975}" \
    --max-model-len "$maxlen" --max-num-batched-tokens 2048 --max-num-seqs "$seqs" \
    --hf-overrides "$QCOVR" --language-model-only --trust-remote-code --moe-backend triton \
    --load-format fastsafetensors \
    ${spec:+--speculative-config "$spec"} >/dev/null
  if ! "$REPO/watch_container.sh" "$name" 1200 8100; then
    echo "--- $tag FAILED，原因 ---"
    docker logs "$name" 2>&1 | grep -oE "(ValueError|RuntimeError|NotImplementedError|AssertionError|AttributeError): .{0,150}" | sort -u | head -3
    docker rm -f "$name" >/dev/null 2>&1; return 1
  fi
  docker logs "$name" 2>&1 | grep -oE "Using [A-Za-z_']+ MoE backend[^.]*\.|GPU KV cache size: [0-9,]+ tokens.*" | tail -2
  docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG -c "python3 -u - <<PY
import json,time,urllib.request
body=json.dumps({'model':'Ornith-$tag','prompt':'Explain in detail why int4 quantization reduces memory bandwidth pressure during decoding.','max_tokens':256,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8100/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=1800)); dt=time.time()-t0
u=d['usage']; print('[$tag] SINGLE-STREAM %.2f tok/s (%d tok in %.1fs)' % (u['completion_tokens']/dt,u['completion_tokens'],dt))
PY"
  curl -s localhost:8100/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{" | sed 's/^/  /'
  [ "$sweep" = "1" ] && docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG \
      -c "python3 -u /work/bench_concurrency.py 8100 Ornith-$tag 1,4,8,16 400 128" 2>&1 | tail -5
  docker rm -f "$name" >/dev/null 2>&1; sleep 25
}

N5='{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_min":5,"prompt_lookup_max":5}'
M1='{"method":"mtp","num_speculative_tokens":1}'
run U3-int4-mtp-256k     262144 16 8 1 "$M1" 1
run U4-int4-mtp-32k      32768  16 8 1 "$M1" 0
run PP8-int4-mtp-256k    262144 16 1 8 "$M1" 0
echo "=== queue done $(date -u +%FT%TZ) ==="
