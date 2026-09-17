#!/usr/bin/env bash
# ============================================================================
# LOCKED PRODUCTION CONFIG -- Ornith-1.5-397B Quark INT8 on 8x MI250X (gfx90a)
#
#   512K context (YaRN 2.0x) + ngram speculative decoding, zero expert offload,
#   CUDA-graph mode.
#
# Why this exact combination (all measured on this box, see LONGCONTEXT_FINDINGS.md):
#   * ngram instead of MTP  -> keeps cross-request PREFIX CACHING alive.
#     vLLM's `use_eagle()` is True only for (eagle, eagle3, mtp, dflash, dspark);
#     with mtp on this hybrid GDN/Mamba model no KV group is annotated as the
#     draft's, so `_warn_if_unannotated_eagle_mamba` flags every group as draft
#     and prefix reuse silently drops to zero (measured: hits 0). With ngram the
#     check returns early: measured hits 232,288 tokens and a repeated 232K-token
#     prefix went 173 s -> 17 s.
#   * 512K (2.0x) instead of 640K (2.45x) -> less RoPE extrapolation risk, and a
#     KV pool that still fits a useful concurrency (581,740 tokens @ seqs 8).
#   * no expert offload -> host-driven expert caches need --enforce-eager, which
#     costs ~4x at batch 1 on this stack (39.96 -> 9.76 tok/s). Not worth it.
#   * YaRN is supported with mrope in this build (rotary_embedding/__init__.py
#     yarn branch keeps mrope_section), so plain --hf-overrides is enough.
#
# Measured with this config: needle retrieval PASSES at 74.7K and 298.9K tokens
# (298.9K is already beyond the model's native 262,144 window), KV pool 581,740
# tokens, short-context decode 15.01 tok/s, prefix reuse 173s -> 17s.
#
# Known costs / caveats:
#   * Long-context prefill degrades sharply: 5,586 tok/s @38K -> ~615 tok/s @299K
#     (a 512K prompt costs on the order of 20 min for the FIRST request).
#     Prefix caching is what makes follow-up questions cheap -- keep ngram on.
#   * Decode at long `max-model-len` is ~15 tok/s vs ~40 tok/s at 32K. The loss
#     comes from max-model-len itself, not from spec decoding; investigating a
#     faster attention path is an open item.
#   * ngram acceptance depends on the workload (good for extraction/quoting/
#     structured output, weak for free-form prose).
# ============================================================================
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-Quark-Int8-Attn}
NAME=${NAME:-ornith-int8-512k}
PORT=${PORT:-8100}
MAX_LEN=${MAX_LEN:-520000}
MEM_UTIL=${MEM_UTIL:-0.975}
MAX_SEQS=${MAX_SEQS:-8}
MAX_BATCHED=${MAX_BATCHED:-2048}
NGRAM_DRAFTS=${NGRAM_DRAFTS:-5}
NGRAM_LOOKUP=${NGRAM_LOOKUP:-5}

# YaRN 2.0x: 262144 (native) -> 524288; mrope_section must be preserved.
HF_OVERRIDES='{"max_position_embeddings":524288,"text_config":{"max_position_embeddings":524288,"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144,"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}}}'
SPEC_CONFIG="{\"method\":\"ngram\",\"num_speculative_tokens\":${NGRAM_DRAFTS},\"prompt_lookup_min\":${NGRAM_LOOKUP},\"prompt_lookup_max\":${NGRAM_LOOKUP}}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
# HBM is not freed instantly when another container dies -> wait for the budget.
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
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 \
  vllm/vllm-openai-rocm:nightly \
    "$MODEL_DIR" \
    --served-model-name Ornith-1.5-397B-Int8-512K \
    --port "$PORT" \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization "$MEM_UTIL" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED" \
    --hf-overrides "$HF_OVERRIDES" \
    --speculative-config "$SPEC_CONFIG" \
    --language-model-only \
    --trust-remote-code \
    --moe-backend triton

echo "serving container : $NAME  (512K + ngram, port $PORT)"
echo "watch logs         : docker logs -f $NAME | grep -E 'KV cache|ERROR|Traceback'"
echo "wait for readiness : ./watch_container.sh $NAME 900 $PORT"
echo "smoke test         : PORT=$PORT MODEL=Ornith-1.5-397B-Int8-512K ./smoke_serve.sh"
echo "prefix-cache check : docker run --rm --network host -v \$PWD:/work vllm/vllm-openai-rocm:nightly \\"
echo "                       -c 'python3 /work/prefix_reuse_probe.py $PORT Ornith-1.5-397B-Int8-512K 200000'"
