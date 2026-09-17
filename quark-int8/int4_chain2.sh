#!/usr/bin/env bash
# CT-Int4 W4A16, fixed: extend the compressed-tensors ignore list with `visual.*`
# (vLLM's internal layer names for the vision tower have no `model.` prefix, so the
# checkpoint's `model.visual.*` pattern did not match and the 576-dim merger got
# int4/group128 -> "input_size_per_partition=576 not divisible by group_size=128").
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8; IMG=vllm/vllm-openai-rocm:nightly
LOG="$REPO/logs/int4_chain2.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
MDL=/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-CT-Int4-W4A16
NGRAM='{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_min":5,"prompt_lookup_max":5}'
QCOVR='{"quantization_config":{"quant_method":"compressed-tensors","format":"pack-quantized","config_groups":{"group_0":{"targets":["Linear"],"input_activations":null,"weights":{"num_bits":4,"type":"int","symmetric":true,"strategy":"group","group_size":128,"dynamic":false,"actorder":null}}},"ignore":["lm_head","model.visual.*","visual.*","re:.*visual.*","mtp.*","*mlp.gate","*mlp.gate.linear","*shared_expert_gate*","*.shared_expert.*","*.linear_attn.*","*.self_attn.*","re:.*embed_tokens.*","re:.*norm.*","re:.*conv.*"],"quantization_status":"compressed"}}'
YARN2='{"max_position_embeddings":524288,"text_config":{"max_position_embeddings":524288,"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144,"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}}}'
YARN4='{"max_position_embeddings":1048576,"text_config":{"max_position_embeddings":1048576,"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144,"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}}}'

wait_gpus () {
  for _ in $(seq 1 240); do
    busy=$(docker ps --format '{{.Names}}' | grep -c '^ornith-' || true)
    free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    [ "$busy" = "0" ] && [ "${free:-0}" -ge 62 ] && return 0
    sleep 15
  done
  return 1
}

launch () {  # tag maxlen seqs util yarn_probe
  local tag=$1 maxlen=$2 seqs=$3 util=$4 yarn=$5
  local name=ornith-$tag
  docker rm -f "$name" >/dev/null 2>&1 || true
  wait_gpus || { echo "GPUs never freed"; return 1; }
  # merge our YARN override with the quantization ignore fix
  local merged; merged=$(python3 - "$yarn" "$QCOVR" <<'PY'
import json, sys
y = json.loads(sys.argv[1]); q = json.loads(sys.argv[2])
y.update(q); print(json.dumps(y))
PY
)
  echo "=== case $tag maxlen=$maxlen seqs=$seqs util=$util $(date -u +%H:%M:%S) ==="
  docker run -d --name "$name" --network host --device /dev/kfd --device /dev/dri --group-add video \
    --shm-size 64G --ulimit memlock=-1:-1 -e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 $IMG \
    $MDL --served-model-name "Ornith-$tag" --port 8100 --tensor-parallel-size 8 \
    --gpu-memory-utilization "$util" --max-model-len "$maxlen" --max-num-batched-tokens 4096 \
    --max-num-seqs "$seqs" --hf-overrides "$merged" --language-model-only --trust-remote-code \
    --moe-backend triton --speculative-config "$NGRAM" >/dev/null
  if ! "$REPO/watch_container.sh" "$name" 1200 8100; then echo "case $tag FAILED"; return 1; fi
  docker logs "$name" 2>&1 | grep -oE "Using [A-Za-z_]+ MoE backend[^.]*\.|Selected [A-Za-z0-9]+ for [A-Za-z0-9]+|GPU KV cache size: [0-9,]+ tokens.*|Available KV cache memory: [0-9.]+ GiB" | tail -4
}

echo "### I1: CT-Int4 512K seqs16 ngram (ignore-fixed)"
if launch I1-int4-512k 520000 16 0.975 "$YARN2"; then
  docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG -c "python3 -u /work/nll_probe.py 8100 Ornith-I1-int4-512k int4-512k" 2>&1 | tail -2
  docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG -c "python3 -u /work/bench_concurrency.py 8100 Ornith-I1-int4-512k 1,4,8,16 400 128" 2>&1 | tail -6
  docker rm -f ornith-I1-int4-512k >/dev/null 2>&1 || true; sleep 15
else
  echo "I1 failed - not running I2"
  exit 1
fi

echo "### I2: CT-Int4 1M context (YaRN 4x) seqs16"
if launch I2-int4-1m 1048576 16 0.975 "$YARN4"; then
  docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG -c "python3 -u /work/bench_concurrency.py 8100 Ornith-I2-int4-1m 1,4,8,16 400 128" 2>&1 | tail -6
fi
echo "=== int4_chain2 done $(date -u +%FT%TZ) ==="
