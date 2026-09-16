#!/usr/bin/env bash
# F2 -> G: the two rounds that actually close the lane question.
#
# Why these two and not a straight "F retry":
#   * vLLM hard-refuses max_model_len > max_position_embeddings (262144) without
#     VLLM_ALLOW_LONG_MAX_MODEL_LEN, so F's 270336 boot was never legal. The
#     256k-class point is instead ISL=237568 against a 262144 server: the InferenceX
#     random dataset overshoots ~8% (283759/262144 observed), and 237568*1.0825+128
#     = 257,295 fits with margin. That is a real ~250k-token single stream.
#   * The real-content probe must run on BOTH arms. On B2 it produced nothing
#     usable: 6/8 requests returned completion_tokens=1 (a truncated file with
#     ignore_eos=false is "done" as far as the model is concerned) and 0
#     SpecDecoding log windows flushed. It now sends a chat-shaped coding-agent
#     turn (system role + repo file + review task) over 12 requests x 512 tokens,
#     which generates continuously and gives the acceptance rate its own evidence.
#   * MTP@250k is measured directly (G); the no-spec@250k number is the only gap
#     and is interpolated from A(1k)=21.57, A(32k)=5.48, C3(128k).
#
# Gated on positive markers; every blocking call has a timeout; the probe runs
# FIRST in each round so a long point cannot starve the decisive measurement.
set -uo pipefail

IX="/home/qiba/ROCm.AI/hyperloom/.cache/InferenceX@3d5581562f643f9bdeb8410cd924e2c70906c966"
OVROOT=/home/qiba/ROCm.AI/hyperloom/patches/fp8-w8a8-emulation-gfx90a
LAB=/home/qiba/ROCm.AI/hyperloom/.tmp/single_stream_lab
MODEL="/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-FP8"
PY=/opt/envs/vllm/bin/python3
LOG="$LAB/sweep.log"

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
busy_dies() {
  rocm-smi --showmeminfo vram 2>/dev/null | awk -F: '/Used Memory/{gsub(/[^0-9]/,"",$2); if($2+0>300000000) c++} END{print c+0}'
}
drain() { for _ in $(seq 1 40); do [[ "$(busy_dies)" == "0" ]] && return 0; sleep 15; done; return 1; }
wait_marker() { # <regex> <max_wait_sec>
  local re="$1" cap="${2:-5400}" waited=0
  while ! grep -aqE "$re" "$LOG" 2>/dev/null; do
    sleep 30; waited=$((waited + 30))
    (( waited > cap )) && { say "GATE TIMEOUT waiting for /$re/"; return 1; }
  done
  return 0
}

run_round() { # run_round <tag> <port> [extra vllm args...]
  local tag="$1" port="$2"; shift 2
  local dir="$LAB/$tag"; mkdir -p "$dir/aot" "$dir/inductor"
  export PYTHONPATH="$OVROOT" VLLM_CACHE_ROOT="$dir/aot" TORCHINDUCTOR_CACHE_DIR="$dir/inductor"
  export VLLM_ROCM_USE_AITER=0 HSA_NO_SCRATCH_RECLAIM=1 PATH="/opt/envs/vllm/bin:$PATH"
  setsid "$PY" /opt/envs/vllm/bin/vllm serve "$MODEL" \
    --port "$port" --tensor-parallel-size=8 --gpu-memory-utilization 0.95 \
    --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 8192 \
    --trust-remote-code --language-model-only --enable-prefix-caching "$@" > "$dir/server.log" 2>&1 &
  echo $! > "$dir/server.pid"
  say "[$tag] booting pid=$(cat $dir/server.pid) extra='$*'"
  local ready=0 i
  for i in $(seq 1 120); do
    sleep 15
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    kill -0 "$(cat $dir/server.pid)" 2>/dev/null || { say "[$tag] DEAD during boot"; break; }
  done
  [[ "$ready" == "1" ]] || { say "[$tag] FAILED to boot"; tail -25 "$dir/server.log" >> "$LOG"; return 1; }
  grep -aiE "Available KV cache memory|GPU KV cache size|Maximum concurrency" "$dir/server.log" \
    | tail -3 | tee -a "$LOG" | cut -c1-165
  say "[$tag] READY"

  # decisive measurement first
  timeout 1500 "$PY" "$LAB/realcode_probe.py" --base-url "http://127.0.0.1:$port" --model "$MODEL" \
      --server-log "$dir/server.log" --out "$dir/realcode_probe.json" \
      --requests 12 --context-tokens 8192 --max-tokens 512 >> "$dir/realcode.log" 2>&1
  say "[$tag] realcode(chat) probe rc=$? -> realcode_probe.json"

  point() { # <conc> <isl> <osl> <n> <label>
    local conc="$1" isl="$2" osl="$3" n="$4" label="$5"
    timeout 2400 env MODEL="$MODEL" CONC="$conc" ISL="$isl" OSL="$osl" RANDOM_RANGE_RATIO=1 \
      NUM_PROMPTS="$n" RESULT_DIR="$dir" RESULT_FILENAME="${tag}_c${conc}_i${isl}_o${osl}_${label}" \
      BENCHMARK_BASE_URL="http://127.0.0.1:$port" MAGPIE_RUN_PHASE=client \
      MAGPIE_TRUST_REMOTE_CODE=1 BENCH_TRUST_REMOTE_CODE=1 MAGPIE_BENCHMARK_PYTHON="$PY" \
      PATH="/opt/envs/vllm/bin:$PATH" bash -c "cd '$IX' && bash benchmarks/vllm_mi250x.sh" \
      >> "$dir/client.log" 2>&1
    say "[$tag] point c$conc i$isl/o$osl n$n rc=$?"
  }
  point 1 1024   1024 3 sanity1k
  point 1 237568 128  2 ctx250k

  local spid pg
  spid=$(cat "$dir/server.pid" 2>/dev/null || true)
  if [[ -n "$spid" ]]; then
    pg=$(ps -o pgid= -p "$spid" 2>/dev/null | tr -d ' ')
    [[ -n "$pg" ]] && kill -TERM -"$pg" 2>/dev/null || true
  fi
  drain || say "[$tag] warning: GPUs still busy after stop"
  say "[$tag] done"
}

wait_marker '\[F_240k\] (done|FAILED|DEAD|GATE)' 60   # F is already dead; just don't hang
run_round F2_nospec 8906 || true
if ! wait_marker '\[F2_nospec\] done' 120; then
  say "chain aborted: F2_nospec incomplete, skipping G"
  exit 1
fi
run_round G_mtp 8907 --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":2}' || true
say "FG chain complete"
