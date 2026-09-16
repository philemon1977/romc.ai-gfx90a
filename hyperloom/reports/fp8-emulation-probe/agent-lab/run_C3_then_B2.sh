#!/usr/bin/env bash
# C3 -> B2, self-deciding and stall-proof.
#
# Gating uses POSITIVE MARKERS in sweep.log (never `pgrep -f <name>`): the C round
# died for 60 minutes because an unreaped zombie ([bgsweep.sh] <defunct>) matched
# its pgrep gate. Every blocking call also carries a `timeout`, so a hung worker
# cannot hold the queue overnight.
#
# Cache flags are chosen from config D's own result, so the pipeline keeps moving
# without an operator awake:
#   D(32k) >= 1.5 x A(32k)  -> the collapse was align-mode state churn -> keep
#                              prefix caching OFF for the long-context ladder
#   otherwise                -> penalty is in the attention kernel -> keep the
#                              lane default (prefix caching ON)
#
# C3 = no speculation, max-model-len 262144 (the trained cap): official points +
#      real-content probe, i.e. the honest single-stream baseline at long context.
# B2 = C3 + MTP (k=2): the same points plus acceptance on real content, which is
#      what decides whether the synthetic 2.00x/2.61x speedup transfers to agents.
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
wait_marker() { # <regex> <max_wait_sec>
  local re="$1" cap="${2:-3600}" waited=0
  while ! grep -aqE "$re" "$LOG" 2>/dev/null; do
    sleep 30; waited=$((waited + 30))
    if (( waited > cap )); then say "GATE TIMEOUT waiting for /$re/ after ${cap}s"; return 1; fi
  done
  return 0
}
drain() { for _ in $(seq 1 40); do [[ "$(busy_dies)" == "0" ]] && return 0; sleep 15; done; return 1; }

pick_flags() {
  "$PY" - "$LAB" <<'PY'
import json, sys
from pathlib import Path
lab = Path(sys.argv[1])
def tps(cfg):
    for f in (lab / cfg).glob("*_c1_i32768_o512_long.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        dur, tok = d.get("duration") or 0, d.get("total_output_tokens") or 0
        if dur and tok:
            return tok / dur
    return None
a, d = tps("A_nospec"), tps("D_nopfx")
print(f"# A(32k)={a and round(a,2)} tok/s  D(32k,no-prefix-cache)={d and round(d,2)} tok/s", file=sys.stderr)
if a and d and d >= 1.5 * a:
    print("--no-enable-prefix-caching")   # align-mode state churn confirmed as the cause
else:
    print("--enable-prefix-caching")      # kernel-bound; keep the lane default
PY
}

run_round() { # run_round <tag> <port> <cache_flags> [extra vllm args...]
  local tag="$1" port="$2" cflags="$3"; shift 3
  local dir="$LAB/$tag"; mkdir -p "$dir/aot" "$dir/inductor"
  export PYTHONPATH="$OVROOT" VLLM_CACHE_ROOT="$dir/aot" TORCHINDUCTOR_CACHE_DIR="$dir/inductor"
  export VLLM_ROCM_USE_AITER=0 HSA_NO_SCRATCH_RECLAIM=1 PATH="/opt/envs/vllm/bin:$PATH"
  setsid "$PY" /opt/envs/vllm/bin/vllm serve "$MODEL" \
    --port "$port" --tensor-parallel-size=8 --gpu-memory-utilization 0.95 \
    --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 8192 \
    --trust-remote-code --language-model-only $cflags "$@" > "$dir/server.log" 2>&1 &
  echo $! > "$dir/server.pid"
  say "[$tag] booting pid=$(cat $dir/server.pid) flags='$cflags $*'"
  local ready=0 i
  for i in $(seq 1 120); do
    sleep 15
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    kill -0 "$(cat $dir/server.pid)" 2>/dev/null || { say "[$tag] DEAD during boot"; break; }
  done
  [[ "$ready" == "1" ]] || { say "[$tag] FAILED to boot"; tail -30 "$dir/server.log" >> "$LOG"; return 1; }
  grep -aiE "Mamba cache mode|Available KV cache memory|GPU KV cache size|Maximum concurrency" \
    "$dir/server.log" | tail -4 | tee -a "$LOG" | cut -c1-170
  say "[$tag] READY"

  point() { # point <conc> <isl> <osl> <n> <label>  -- hard 40 min cap per point
    local conc="$1" isl="$2" osl="$3" n="$4" label="$5"
    timeout 2400 env MODEL="$MODEL" CONC="$conc" ISL="$isl" OSL="$osl" RANDOM_RANGE_RATIO=1 \
      NUM_PROMPTS="$n" RESULT_DIR="$dir" RESULT_FILENAME="${tag}_c${conc}_i${isl}_o${osl}_${label}" \
      BENCHMARK_BASE_URL="http://127.0.0.1:$port" MAGPIE_RUN_PHASE=client \
      MAGPIE_TRUST_REMOTE_CODE=1 BENCH_TRUST_REMOTE_CODE=1 MAGPIE_BENCHMARK_PYTHON="$PY" \
      PATH="/opt/envs/vllm/bin:$PATH" bash -c "cd '$IX' && bash benchmarks/vllm_mi250x.sh" \
      >> "$dir/client.log" 2>&1
    say "[$tag] point c$conc i$isl/o$osl n$n rc=$?"
  }

  point 1 1024 1024 5 base
  point 1 32768 512 3 long
  timeout 1500 "$PY" "$LAB/realcode_probe.py" --base-url "http://127.0.0.1:$port" --model "$MODEL" \
      --server-log "$dir/server.log" --out "$dir/realcode_probe.json" \
      --requests 8 --context-tokens 4096 --max-tokens 256 >> "$dir/realcode.log" 2>&1
  say "[$tag] realcode probe rc=$? -> realcode_probe.json"
  point 1 131072 256 2 mid128k
  point 1 262144 256 2 max256k

  local spid pg
  spid=$(cat "$dir/server.pid" 2>/dev/null || true)
  if [[ -n "$spid" ]]; then
    pg=$(ps -o pgid= -p "$spid" 2>/dev/null | tr -d ' ')
    [[ -n "$pg" ]] && kill -TERM -"$pg" 2>/dev/null || true
  fi
  drain || say "[$tag] warning: GPUs still busy after stop"
  say "[$tag] done"
}

# ---- chain ------------------------------------------------------------------
wait_marker '\[D\] (done|FAILED|DEAD)' 5400 || { say "chain aborted: D never reported"; exit 1; }
FLAGS=$(pick_flags); say "chain: cache flags -> $FLAGS"

run_round C3_nospec 8903 "$FLAGS" || true
if ! wait_marker '\[C3_nospec\] done' 120; then
  say "chain aborted: C3_nospec did not complete, so B2_mtp would only burn another boot"
  exit 1
fi
run_round B2_mtp 8904 "$FLAGS" --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":2}' || true
say "chain complete"
