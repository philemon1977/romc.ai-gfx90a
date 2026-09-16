#!/usr/bin/env bash
# F -- the corrected 256k-class single-stream point (no speculation).
#
# Why C3's max256k point failed 100% of requests: InferenceX's random dataset
# overshoots the requested length by ~8% ("Token indices sequence length is longer
# than the specified maximum sequence length for this model (283759 > 262144)"), so
# asking for ISL == max-model-len is unservable. Fix: ask for 245760 (245760 x 1.0825
# = 266050) against a server allowed 270336 tokens. That is a true 240k-class
# single-stream measurement instead of a client-side rejection.
#
# Gated on the POSITIVE MARKER "[B2_mtp] done" (never pgrep name patterns).
set -uo pipefail

IX="/home/qiba/ROCm.AI/hyperloom/.cache/InferenceX@3d5581562f643f9bdeb8410cd924e2c70906c966"
OVROOT=/home/qiba/ROCm.AI/hyperloom/patches/fp8-w8a8-emulation-gfx90a
LAB=/home/qiba/ROCm.AI/hyperloom/.tmp/single_stream_lab
MODEL="/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-FP8"
PY=/opt/envs/vllm/bin/python3
LOG="$LAB/sweep.log"
TAG=F_240k; PORT=8905
DIR="$LAB/$TAG"

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
busy_dies() {
  rocm-smi --showmeminfo vram 2>/dev/null | awk -F: '/Used Memory/{gsub(/[^0-9]/,"",$2); if($2+0>300000000) c++} END{print c+0}'
}

say "[$TAG] armed (waiting for the B2_mtp completion marker)"
waited=0
while ! grep -aqE '\[B2_mtp\] done' "$LOG" 2>/dev/null; do
  sleep 30; waited=$((waited + 30))
  (( waited > 5400 )) && { say "[$TAG] GATE TIMEOUT after 90 min -- aborting"; exit 1; }
done
for _ in $(seq 1 40); do [[ "$(busy_dies)" == "0" ]] && break; sleep 15; done
say "[$TAG] GPUs free, booting 240k-capable config"

mkdir -p "$DIR/aot" "$DIR/inductor"
export PYTHONPATH="$OVROOT" VLLM_CACHE_ROOT="$DIR/aot" TORCHINDUCTOR_CACHE_DIR="$DIR/inductor"
export VLLM_ROCM_USE_AITER=0 HSA_NO_SCRATCH_RECLAIM=1 PATH="/opt/envs/vllm/bin:$PATH"
setsid "$PY" /opt/envs/vllm/bin/vllm serve "$MODEL" \
  --port "$PORT" --tensor-parallel-size=8 --gpu-memory-utilization 0.95 \
  --max-model-len 270336 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --trust-remote-code --language-model-only --enable-prefix-caching \
  > "$DIR/server.log" 2>&1 &
echo $! > "$DIR/server.pid"
say "[$TAG] booting pid=$(cat $DIR/server.pid) max-len=270336"

ready=0
for _ in $(seq 1 120); do
  sleep 15
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 "$(cat $DIR/server.pid)" 2>/dev/null || { say "[$TAG] DEAD during boot"; break; }
done
[[ "$ready" == "1" ]] || { say "[$TAG] FAILED to boot"; tail -30 "$DIR/server.log" >> "$LOG"; exit 1; }
grep -aiE "Available KV cache memory|GPU KV cache size|Maximum concurrency" "$DIR/server.log" \
  | tail -3 | tee -a "$LOG" | cut -c1-170
say "[$TAG] READY"

point() { # <conc> <isl> <osl> <n> <label>
  local conc="$1" isl="$2" osl="$3" n="$4" label="$5"
  timeout 2400 env MODEL="$MODEL" CONC="$conc" ISL="$isl" OSL="$osl" RANDOM_RANGE_RATIO=1 \
    NUM_PROMPTS="$n" RESULT_DIR="$DIR" RESULT_FILENAME="${TAG}_c${conc}_i${isl}_o${osl}_${label}" \
    BENCHMARK_BASE_URL="http://127.0.0.1:$PORT" MAGPIE_RUN_PHASE=client \
    MAGPIE_TRUST_REMOTE_CODE=1 BENCH_TRUST_REMOTE_CODE=1 MAGPIE_BENCHMARK_PYTHON="$PY" \
    PATH="/opt/envs/vllm/bin:$PATH" bash -c "cd '$IX' && bash benchmarks/vllm_mi250x.sh" \
    >> "$DIR/client.log" 2>&1
  say "[$TAG] point c$conc i$isl/o$osl n$n rc=$?"
}

point 1 1024   1024 3 sanity1k
point 1 131072 512  2 ctx128k
point 1 245760 512  2 ctx240k
timeout 1500 "$PY" "$LAB/realcode_probe.py" --base-url "http://127.0.0.1:$PORT" --model "$MODEL" \
    --server-log "$DIR/server.log" --out "$DIR/realcode_probe.json" \
    --requests 8 --context-tokens 8192 --max-tokens 256 >> "$DIR/realcode.log" 2>&1
say "[$TAG] realcode probe rc=$?"

SPID=$(cat "$DIR/server.pid" 2>/dev/null || true)
if [[ -n "$SPID" ]]; then
  PG=$(ps -o pgid= -p "$SPID" 2>/dev/null | tr -d ' ')
  [[ -n "$PG" ]] && kill -TERM -"$PG" 2>/dev/null || true
fi
for _ in $(seq 1 40); do [[ "$(busy_dies)" == "0" ]] && break; sleep 15; done
say "[$TAG] done"
