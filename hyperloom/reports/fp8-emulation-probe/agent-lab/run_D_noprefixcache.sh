#!/usr/bin/env bash
# Config D -- isolate the long-context decode collapse.
#
# Measured (official client, CONC=1, same server otherwise):
#   ISL 1024  -> decode 21.57 tok/s, TPOT 45.16 ms
#   ISL 32768 -> decode  5.48 tok/s, TPOT 170.78 ms   (+125.6 ms/token)
# The extra 125 ms/token cannot be physics: at 16.0 KiB/token/die of KV, a 32k
# context is ~0.5 GiB of KV per die per token, which is ~0.4 ms at HBM peak, and
# the attention FLOPs for 15 full-attention layers are ~8 GFLOP/token (<1% of the
# 8-die BF16 peak). So the marginal cost is per-step work that scales with context,
# and the prime suspect is the hybrid state path: vLLM auto-set
# "Mamba cache mode is set to 'align' ... when prefix caching is enabled" for this
# checkpoint, and align-mode state realignment is the only mechanism here that
# grows with sequence length.
#
# D therefore re-runs the two decisive points with prefix caching OFF. If decode at
# 32k recovers toward ~20 tok/s, the collapse is align-mode/prefix-caching, and the
# agent lane's biggest lever stops being kernels and becomes a cache-mode decision
# (plus keeping prefix hits via a cheaper path). If it stays ~5 tok/s, it is the
# paged-attention kernel on gfx90a and belongs to the kernel agent.
set -uo pipefail

IX="/home/qiba/ROCm.AI/hyperloom/.cache/InferenceX@3d5581562f643f9bdeb8410cd924e2c70906c966"
OVROOT=/home/qiba/ROCm.AI/hyperloom/patches/fp8-w8a8-emulation-gfx90a
LAB=/home/qiba/ROCm.AI/hyperloom/.tmp/single_stream_lab
MODEL="/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-FP8"
PY=/opt/envs/vllm/bin/python3
LOG="$LAB/sweep.log"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
vram_free() {
  rocm-smi --showmeminfo vram 2>/dev/null | awk -F: '/Used Memory/{gsub(/[^0-9]/,"",$2); if($2+0>300000000) c++} END{exit (c+0==0)?0:1}'
}

say "[D] armed (waiting for the C_256k round to finish)"
while pgrep -f "run_C_256[k].sh" >/dev/null 2>&1; do sleep 30; done
for _ in $(seq 1 40); do vram_free && break; sleep 15; done
say "[D] GPUs free, booting no-prefix-cache config"

TAG=D_nopfx; PORT=8902; DIR="$LAB/$TAG"; mkdir -p "$DIR/aot" "$DIR/inductor"
export PYTHONPATH="$OVROOT" VLLM_CACHE_ROOT="$DIR/aot" TORCHINDUCTOR_CACHE_DIR="$DIR/inductor"
export VLLM_ROCM_USE_AITER=0 HSA_NO_SCRATCH_RECLAIM=1 PATH="/opt/envs/vllm/bin:$PATH"
# identical to A except prefix caching is explicitly off, so mamba_cache_mode
# falls back to its default "none" instead of the "align" mode prefix caching
# forces. Verified flags: --no-enable-prefix-caching, --mamba-cache-mode {all,
# align,none}. If decode recovers, the follow-up is --mamba-cache-mode all
# (prefix hits without the align realignment path).
setsid "$PY" /opt/envs/vllm/bin/vllm serve "$MODEL" \
  --port "$PORT" --tensor-parallel-size=8 --gpu-memory-utilization 0.95 \
  --max-model-len 65536 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --trust-remote-code --language-model-only --no-enable-prefix-caching \
  > "$DIR/server.log" 2>&1 &
echo $! > "$DIR/server.pid"
say "[D] booting pid=$(cat $DIR/server.pid)"

ready=0
for _ in $(seq 1 140); do
  sleep 15
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "$(cat $DIR/server.pid)" 2>/dev/null || { say "[D] DEAD during boot"; break; }
done
[[ "$ready" == "1" ]] || { say "[D] FAILED to boot"; tail -30 "$DIR/server.log" >> "$LOG"; exit 1; }
grep -aiE "Mamba cache mode|Available KV cache memory|GPU KV cache size" "$DIR/server.log" | tail -3 | tee -a "$LOG" | cut -c1-170
say "[D] READY"

point() { # point <conc> <isl> <osl> <num_prompts> <label>
  local conc="$1" isl="$2" osl="$3" n="$4" label="$5"
  ( cd "$IX" || exit 1
    MODEL="$MODEL" CONC="$conc" ISL="$isl" OSL="$osl" RANDOM_RANGE_RATIO=1 \
    NUM_PROMPTS="$n" RESULT_DIR="$DIR" RESULT_FILENAME="${TAG}_c${conc}_i${isl}_o${osl}_${label}" \
    BENCHMARK_BASE_URL="http://127.0.0.1:$PORT" MAGPIE_RUN_PHASE=client \
    MAGPIE_TRUST_REMOTE_CODE=1 BENCH_TRUST_REMOTE_CODE=1 \
    MAGPIE_BENCHMARK_PYTHON="$PY" PATH="/opt/envs/vllm/bin:$PATH" \
    bash "$IX/benchmarks/vllm_mi250x.sh"
  ) >> "$DIR/client.log" 2>&1
  say "[D] point c$conc i$isl/o$osl n$n rc=$?"
}

point 1 1024  1024 5 base      # must match A's 21.57 tok/s if only the cache mode changed
point 1 32768 512  3 long      # the decisive number: recovered or not
point 16 1024 512  32 env16
SPID=$(cat "$DIR/server.pid" 2>/dev/null || true)
if [[ -n "$SPID" ]]; then
  PG=$(ps -o pgid= -p "$SPID" 2>/dev/null | tr -d ' ')
  [[ -n "$PG" ]] && kill -TERM -"$PG" 2>/dev/null || true
fi
for _ in $(seq 1 24); do sleep 5; vram_free && break; done
say "[D] done, server stopped"
