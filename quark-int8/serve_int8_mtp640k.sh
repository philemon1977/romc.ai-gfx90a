#!/usr/bin/env bash
# ============================================================================
# ALTERNATIVE CONFIG -- 640K context + MTP speculative decoding.
#
# Use ONLY when every request carries a brand-new document (single-shot), so the
# loss of cross-request prefix caching does not matter. Otherwise prefer
# ./serve_int8.sh (512K + ngram), which keeps prefix caching alive.
#
# Why prefix caching dies here: with method=mtp, vLLM's `use_eagle()` is True and
# on this hybrid GDN/Mamba model no KV cache group is annotated as the draft
# model's, so `_warn_if_unannotated_eagle_mamba` treats every group (incl. Mamba
# groups) as a draft group. A Mamba group cannot satisfy the widened lookup
# window, so cross-request reuse drops to zero *silently* (measured: prefix_cache
# hits = 0 while queries = 74,761). Any external KV offload tier would likewise
# store without ever serving a hit.
#
# Measured with this config: KV pool 644,713 tokens, but "Maximum concurrency for
# 640,000 tokens per request: 1.01x" -> a full 640K prompt leaves only ~4.7K
# tokens for generation. Keep MAX_LEN around 600-620K for real use (prompt +
# output share the pool). Short-context decode 12.26 tok/s.
# ============================================================================
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-Quark-Int8-Attn}
NAME=${NAME:-ornith-int8-640k-mtp}
PORT=${PORT:-8100}
MAX_LEN=${MAX_LEN:-620000}
MEM_UTIL=${MEM_UTIL:-0.975}
MAX_SEQS=${MAX_SEQS:-2}
MTP_TOKENS=${MTP_TOKENS:-1}

HF_OVERRIDES='{"max_position_embeddings":640000,"text_config":{"max_position_embeddings":640000,"rope_parameters":{"rope_type":"yarn","factor":2.45,"original_max_position_embeddings":262144,"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}}}'

docker rm -f "$NAME" >/dev/null 2>&1 || true
NEED_GIB=$(python3 -c "print(int(${MEM_UTIL}*63.98)+1)")
for _ in $(seq 1 90); do
  free_gib=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 \
             | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "${free_gib:-0}" -ge "$NEED_GIB" ] && break
  sleep 10
done

docker run -d --name "$NAME" --network host \
  --device /dev/kfd --device /dev/dri --group-add video \
  --shm-size 64G --ulimit memlock=-1:-1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 \
  vllm/vllm-openai-rocm:nightly \
    "$MODEL_DIR" \
    --served-model-name Ornith-1.5-397B-Int8-640K \
    --port "$PORT" --tensor-parallel-size 8 \
    --gpu-memory-utilization "$MEM_UTIL" --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens 2048 \
    --hf-overrides "$HF_OVERRIDES" \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}" \
    --language-model-only --trust-remote-code --moe-backend triton

echo "serving container: $NAME (640K + MTP; prefix caching will be DISABLED)"
