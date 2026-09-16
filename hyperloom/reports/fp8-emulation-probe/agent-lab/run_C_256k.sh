#!/usr/bin/env bash
# Config C -- the 256k single-stream point for the agent lane.
#
# Why this is a tuning measurement, not the capacity phase: with
# --max-num-seqs 16 the emulation server already reserves a 452,748-token KV pool
# (measured: 6.9 GiB/die at 16.0 KiB/token/die), and the model's trained context
# is 262,144 -- so ONE 256k stream fits today. What is unknown is what it costs:
# prefill TTFT at 256k, and whether decode slows down as the 15 full-attention
# layers accumulate KV (45 of 60 layers are GDN linear attention = constant state).
#
# Armed after sweep.sh (A/B) finishes; needs its own boot because max-model-len is
# a boot-time knob.
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

say "[C] armed (waiting for sweep.sh A/B to finish)"
while pgrep -f "swee[p].sh" >/dev/null 2>&1; do sleep 30; done
for _ in $(seq 1 40); do vram_free && break; sleep 15; done
say "[C] GPUs free, booting 256k config"

TAG=C_256k; PORT=8901; DIR="$LAB/$TAG"; mkdir -p "$DIR/aot" "$DIR/inductor"
export PYTHONPATH="$OVROOT" VLLM_CACHE_ROOT="$DIR/aot" TORCHINDUCTOR_CACHE_DIR="$DIR/inductor"
export VLLM_ROCM_USE_AITER=0 HSA_NO_SCRATCH_RECLAIM=1 PATH="/opt/envs/vllm/bin:$PATH"
setsid "$PY" /opt/envs/vllm/bin/vllm serve "$MODEL" \
  --port "$PORT" --tensor-parallel-size=8 --gpu-memory-utilization 0.95 \
  --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --trust-remote-code --language-model-only --enable-prefix-caching \
  > "$DIR/server.log" 2>&1 &
echo $! > "$DIR/server.pid"
say "[C] booting pid=$(cat $DIR/server.pid) max-len=262144"

ready=0
for _ in $(seq 1 140); do
  sleep 15
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "$(cat $DIR/server.pid)" 2>/dev/null || { say "[C] DEAD during boot"; break; }
done
[[ "$ready" == "1" ]] || { say "[C] FAILED to boot -- 256k needs the capacity phase first"; tail -30 "$DIR/server.log" >> "$LOG"; exit 1; }
grep -aiE "Available KV cache memory|GPU KV cache size|Maximum concurrency" "$DIR/server.log" | tail -3 | tee -a "$LOG" | cut -c1-170
say "[C] READY"

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
  say "[C] point c$conc i$isl/o$osl n$n rc=$?"
}

# context ladder at one stream: what does 256k cost vs 32k/128k
point 1 32768  1024 3 ref32k
point 1 131072 1024 2 mid128k
point 1 262144 1024 2 max256k
# (no 16 x 256k point: 16 x 262,144 tokens = 4.2M of KV against a 452,748-token
# pool, so the pool math already answers it -- the ceiling is ~28,297 tokens per
# session at 16 streams. Measuring it would cost ~105 min of prefill.)
"$PY" "$LAB/prefix_probe.py" --base-url "http://127.0.0.1:$PORT" --model "$MODEL" \
    --out "$DIR/prefix_probe.json" --context 131072 --turns 3 --osl 256 >> "$DIR/prefix.log" 2>&1
say "[C] prefix probe rc=$? (131k shared prefix, 3 turns)"
SPID=$(cat "$DIR/server.pid" 2>/dev/null || true)
if [[ -n "$SPID" ]]; then
  PG=$(ps -o pgid= -p "$SPID" 2>/dev/null | tr -d ' ')
  if [[ -n "$PG" ]]; then kill -TERM -"$PG" 2>/dev/null || true; fi
fi
for _ in $(seq 1 24); do sleep 5; vram_free && break; done
say "[C] done, server stopped"
