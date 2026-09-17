#!/usr/bin/env bash
# PP8 arm (single-stream focus): int4 + MTP(1), TP=1 x PP=8, 256K, seqs16, graph mode.
# Fast start: persistent Triton cache + fastsafetensors. Directly comparable to U3
# (int4 + MTP + TP8 @256K = 38.17 tok/s single-stream, KV 1,853,658 tokens).
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8; IMG=vllm/vllm-openai-rocm:nightly
LOG="$REPO/logs/pp8.log"; mkdir -p "$REPO/logs" "$REPO/triton_cache"
exec > >(tee -a "$LOG") 2>&1
MDL=/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-CT-Int4-W4A16
QCOVR='{"quantization_config":{"quant_method":"compressed-tensors","format":"pack-quantized","config_groups":{"group_0":{"targets":["Linear"],"input_activations":null,"weights":{"num_bits":4,"type":"int","symmetric":true,"strategy":"group","group_size":128,"dynamic":false,"actorder":null}}},"ignore":["re:^lm_head","re:.*visual\\..*","re:^mtp\\..*","re:.*mlp\\.gate$","re:.*mlp\\.gate\\.linear$","re:.*shared_expert_gate.*","re:.*\\.shared_expert\\..*","re:.*\\.linear_attn\\..*","re:.*\\.self_attn\\..*","re:.*embed_tokens.*","re:.*norm.*","re:.*conv.*"],"quantization_status":"compressed"}}'
NAME=ornith-PP8-int4-mtp
docker rm -f $NAME >/dev/null 2>&1 || true
for c in $(docker ps -a --format '{{.Names}}' | grep '^ornith-' || true); do docker rm -f "$c" >/dev/null 2>&1; done
for _ in $(seq 1 120); do
  alive=$(docker ps -a --format '{{.Names}}|{{.Status}}' | grep '^ornith-' | grep -c Up || true)
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "$alive" = "0" ] && [ "${free:-0}" -ge 62 ] && break; sleep 10
done
echo "=== PP8: TP=1 PP=8, int4+MTP(1), maxlen 262144, seqs16, graph, fastsafetensors $(date -u +%H:%M:%S) ==="
docker run -d --name $NAME --network host --device /dev/kfd --device /dev/dri --group-add video \
  --shm-size 64G --ulimit memlock=-1:-1 -e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e TRITON_CACHE_DIR=/triton_cache \
  -v "$REPO/triton_cache:/triton_cache" -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 $IMG \
  $MDL --served-model-name Ornith-PP8 --port 8100 --tensor-parallel-size 1 \
  --pipeline-parallel-size 8 --gpu-memory-utilization 0.975 --max-model-len 262144 \
  --max-num-batched-tokens 2048 --max-num-seqs 16 --hf-overrides "$QCOVR" --language-model-only \
  --trust-remote-code --moe-backend triton --load-format fastsafetensors \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' >/dev/null
if ! "$REPO/watch_container.sh" $NAME 1200 8100; then
  echo "PP8 FAILED — 真实原因："
  docker logs $NAME 2>&1 | grep -oE "(ValueError|RuntimeError|NotImplementedError|AssertionError|AttributeError): .{0,170}" | sort -u | head -5
  docker rm -f $NAME >/dev/null 2>&1; exit 1
fi
docker logs $NAME 2>&1 | grep -oE "Using [A-Za-z_']+ MoE backend[^.]*\.|Loading weights took [0-9.]+ seconds|GPU KV cache size: [0-9,]+ tokens.*" | tail -3
docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG -c "python3 -u - <<'PY'
import json,time,urllib.request
body=json.dumps({'model':'Ornith-PP8','prompt':'Explain in detail why int4 quantization reduces memory bandwidth pressure during decoding.','max_tokens':256,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8100/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=1800)); dt=time.time()-t0
u=d['usage']; print('[PP8-int4-mtp] SINGLE-STREAM %.2f tok/s (%d tok in %.1fs)' % (u['completion_tokens']/dt,u['completion_tokens'],dt))
PY"
curl -s localhost:8100/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{" | sed 's/^/  /'
docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG \
  -c "python3 -u /work/bench_concurrency.py 8100 Ornith-PP8 1,4,8,16 400 128" 2>&1 | tail -5
echo "=== PP8 done $(date -u +%FT%TZ) ==="
