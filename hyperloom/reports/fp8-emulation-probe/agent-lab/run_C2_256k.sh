#!/usr/bin/env bash
# Config C2 -- the 256k single-stream ladder, with the cache flags chosen from
# what config D says.
#
# Why the original C never ran (and why this one waits differently): its
# readiness gate was `while pgrep -f "swee[p].sh"`. The container shares the host
# PID namespace and holds an unreaped zombie, pid 9772 ZN `[bgsweep.sh] <defunct>`
# (ppid 1, never reaped), whose command name CONTAINS the literal string
# "sweep.sh" -- so the pattern matched forever and C waited 60 minutes while the
# GPUs sat idle. The bracket trick only avoids self-matching by the grep itself;
# it does not help against zombies. Never gate on a name pattern here: gate on
# real GPU occupancy and on an explicit server-PID absence.
#
# Cost control: decode at 256k is what we are measuring, so OSL is capped at 256
# tokens (TPOT is OSL-insensitive) and n=2 per point. Worst case (if D shows the
# long-context penalty is the paged-attention kernel, ~1.05 s/token at 256k) this
# round is ~45 min; if D shows it was align-mode state churn, ~25 min.
set -uo pipefail

IX="/home/qiba/ROCm.AI/hyperloom/.cache/InferenceX@3d5581562f643f9bdeb8410cd924e2c70906c966"
OVROOT=/home/qiba/ROCm.AI/hyperloom/patches/fp8-w8a8-emulation-gfx90a
LAB=/home/qiba/ROCm.AI/hyperloom/.tmp/single_stream_lab
MODEL="/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-FP8"
PY=/opt/envs/vllm/bin/python3
LOG="$LAB/sweep.log"
# set at arm time: "--enable-prefix-caching" (align) / "--no-enable-prefix-caching" (none)
# / "--no-enable-prefix-caching --mamba-cache-mode all" etc.
CACHE_FLAGS="${CACHE_FLAGS:---enable-prefix-caching}"
TAG="${TAG:-C2_256k}"; PORT="${PORT:-8903}"

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
busy_dies() {
  rocm-smi --showmeminfo vram 2>/dev/null | awk -F: '/Used Memory/{gsub(/[^0-9]/,"",$2); if($2+0>300000000) c++} END{print c+0}'
}
server_running() { pgrep -f "vllm serv[e] $MODEL" | grep -v "^$" | head -1; }

say "[$TAG] armed (cache flags: $CACHE_FLAGS)"
for _ in $(seq 1 240); do
  [[ "$(busy_dies)" == "0" ]] && [[ -z "$(server_running)" ]] && break
  sleep 15
done
[[ "$(busy_dies)" == "0" ]] || { say "[$TAG] GPUs still busy after 60 min -- aborting"; exit 1; }
say "[$TAG] GPUs free, booting 256k config"

DIR="$LAB/$TAG"; mkdir -p "$DIR/aot" "$DIR/inductor"
export PYTHONPATH="$OVROOT" VLLM_CACHE_ROOT="$DIR/aot" TORCHINDUCTOR_CACHE_DIR="$DIR/inductor"
export VLLM_ROCM_USE_AITER=0 HSA_NO_SCRATCH_RECLAIM=1 PATH="/opt/envs/vllm/bin:$PATH"
setsid "$PY" /opt/envs/vllm/bin/vllm serve "$MODEL" \
  --port "$PORT" --tensor-parallel-size=8 --gpu-memory-utilization 0.95 \
  --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --trust-remote-code --language-model-only $CACHE_FLAGS \
  > "$DIR/server.log" 2>&1 &
echo $! > "$DIR/server.pid"
say "[$TAG] booting pid=$(cat $DIR/server.pid) max-len=262144"

ready=0
for _ in $(seq 1 140); do
  sleep 15
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "$(cat $DIR/server.pid)" 2>/dev/null || { say "[$TAG] DEAD during boot"; break; }
done
[[ "$ready" == "1" ]] || { say "[$TAG] FAILED to boot"; tail -30 "$DIR/server.log" >> "$LOG"; exit 1; }
grep -aiE "Mamba cache mode|Available KV cache memory|GPU KV cache size|Maximum concurrency" \
  "$DIR/server.log" | tail -4 | tee -a "$LOG" | cut -c1-170
say "[$TAG] READY"

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
  say "[$TAG] point c$conc i$isl/o$osl n$n rc=$?"
}

point 1 32768  512  3 ref32k
point 1 131072 256  2 mid128k
point 1 262144 256  2 max256k
"$PY" "$LAB/prefix_probe.py" --base-url "http://127.0.0.1:$PORT" --model "$MODEL" \
    --out "$DIR/prefix_probe.json" --context 131072 --turns 3 --osl 256 >> "$DIR/prefix.log" 2>&1
say "[$TAG] prefix probe rc=$? (131k shared prefix, 3 turns)"

SPID=$(cat "$DIR/server.pid" 2>/dev/null || true)
if [[ -n "$SPID" ]]; then
  PG=$(ps -o pgid= -p "$SPID" 2>/dev/null | tr -d ' ')
  [[ -n "$PG" ]] && kill -TERM -"$PG" 2>/dev/null || true
fi
for _ in $(seq 1 24); do sleep 5; [[ "$(busy_dies)" == "0" ]] && break; done
say "[$TAG] done, server stopped"
