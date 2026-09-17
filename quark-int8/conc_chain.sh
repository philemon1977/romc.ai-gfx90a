#!/usr/bin/env bash
# Concurrency sweep (1/4/8/16) on three configs, plus the layout lever test.
#   1) locked 512K + ngram, seqs 16
#   2) same + VLLM_KV_CACHE_LAYOUT=NHD
#   3) 32K + ngram, seqs 16 (isolates the max-model-len effect on decode speed)
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8; IMG=vllm/vllm-openai-rocm:nightly
LOG="$REPO/logs/conc_chain.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
MDL=/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-Quark-Int8-Attn
NGRAM='{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_min":5,"prompt_lookup_max":5}'
YARN='{"max_position_embeddings":524288,"text_config":{"max_position_embeddings":524288,"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144,"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}}}'

run_case () {  # tag maxlen seqs util yarn extra_env
  local tag=$1 maxlen=$2 seqs=$3 util=$4 yarn=$5 extra=$6
  local name=ornith-$tag
  docker rm -f "$name" >/dev/null 2>&1 || true
  local need; need=$(python3 -c "print(int(${util}*63.98)+1)")
  for _ in $(seq 1 90); do
    f=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    [ "${f:-0}" -ge "$need" ] && break; sleep 10
  done
  local yarnarg=(); [ "$yarn" = "1" ] && yarnarg=(--hf-overrides "$YARN")
  echo "=== case $tag maxlen=$maxlen seqs=$seqs util=$util yarn=$yarn extra='$extra' $(date -u +%H:%M:%S) ==="
  docker run -d --name "$name" --network host --device /dev/kfd --device /dev/dri --group-add video \
    --shm-size 64G --ulimit memlock=-1:-1 -e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 $extra -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 $IMG \
    $MDL --served-model-name "Ornith-$tag" --port 8100 --tensor-parallel-size 8 \
    --gpu-memory-utilization "$util" --max-model-len "$maxlen" --max-num-batched-tokens 2048 \
    --max-num-seqs "$seqs" "${yarnarg[@]}" --language-model-only --trust-remote-code \
    --moe-backend triton --speculative-config "$NGRAM" >/dev/null
  if ! "$REPO/watch_container.sh" "$name" 900 8100; then echo "case $tag FAILED -> skipping"; return 1; fi
  docker logs "$name" 2>&1 | grep -oE "kv cache group sizes \[[0-9, ]+\]|kv lcm block sizes [0-9]+|GPU KV cache size: [0-9,]+ tokens.*" | tail -2
  docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG \
    -c "python3 -u /work/bench_concurrency.py 8100 Ornith-$tag 1,4,8,16 400 128" 2>&1 | tail -6
  docker rm -f "$name" >/dev/null 2>&1 || true
  sleep 15
}

run_case C1-512k-s16 520000 16 0.975 1 ""
run_case C2-512k-nhd 520000 16 0.975 1 "-e VLLM_KV_CACHE_LAYOUT=NHD"
run_case C3-32k-s16  32768  16 0.96  0 ""
echo "=== conc_chain done $(date -u +%FT%TZ) ==="
