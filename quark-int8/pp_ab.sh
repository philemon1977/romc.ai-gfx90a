#!/usr/bin/env bash
# PP vs TP A/B for single-stream TPS: int4 + MTP(1) + PP8 (TP=1), everything else
# identical to the TP8 int4 arm. Also reports the KV pool (expect ~8x per rank).
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8; IMG=vllm/vllm-openai-rocm:nightly
LOG="$REPO/logs/pp_ab.log"; mkdir -p "$REPO/logs"
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
NAME=ornith-PP8-int4-mtp
docker rm -f $NAME >/dev/null 2>&1 || true
wait_gpus || { echo "GPUs busy"; exit 1; }
echo "=== PP8 arm: int4 + MTP(1), TP=1 PP=8, maxlen 262144, seqs 16 $(date -u +%H:%M:%S) ==="
docker run -d --name $NAME --network host --device /dev/kfd --device /dev/dri --group-add video \
  --shm-size 64G --ulimit memlock=-1:-1 -e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 $IMG \
  $MDL --served-model-name Ornith-PP8 --port 8100 --tensor-parallel-size 1 \
  --pipeline-parallel-size 8 --gpu-memory-utilization 0.975 --max-model-len 262144 \
  --max-num-batched-tokens 2048 --max-num-seqs 16 --hf-overrides "$QCOVR" --language-model-only \
  --trust-remote-code --moe-backend triton \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' >/dev/null
if ! "$REPO/watch_container.sh" $NAME 1200 8100; then
  echo "PP8 arm FAILED — 记录原因："
  docker logs $NAME 2>&1 | grep -oE "(ValueError|RuntimeError|NotImplementedError|AssertionError): .{0,150}" | sort -u | head -4
  exit 1
fi
docker logs $NAME 2>&1 | grep -oE "Using [A-Za-z_]+ MoE backend[^.]*\.|GPU KV cache size: [0-9,]+ tokens.*|pipeline_parallel_size=[0-9]+" | tail -3
docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG -c "python3 -u - <<'PY'
import json,time,urllib.request
body=json.dumps({'model':'Ornith-PP8','prompt':'Explain in detail why int4 quantization reduces memory bandwidth pressure during decoding.','max_tokens':256,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8100/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=1800)); dt=time.time()-t0
u=d['usage']; print('[PP8-int4-mtp] SINGLE-STREAM %.2f tok/s (%d tok in %.1fs)' % (u['completion_tokens']/dt,u['completion_tokens'],dt))
PY"
curl -s localhost:8100/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{" | sed 's/^/  /'
echo "=== PP8 done $(date -u +%FT%TZ) ==="
