#!/bin/bash
# Wait for the in-flight Hyperloom run to close, then measure the gfx90a
# FP8->BF16 dequant-emulation route (single stream + conc 8/32).
#
# Why it waits: a live run reaps foreign GPU processes (2026-09-15 18:16:16,
# specialist 8628519056... ran `kill -TERM -35120` on an earlier standalone
# probe of ours, logged as "freed 8 GPUs from leaked probe"), so a side-probe
# can only coexist with nobody.
#
# Everything is logged to autoprobe.log and the result is written to
# probe_results.json / PROBE_SUMMARY.md in this directory.
set -u

MY=/home/qiba/ROCm.AI/hyperloom/.tmp/emulation_autoprobe
SES=/home/qiba/ROCm.AI/hyperloom/session/Ornith-1.5-397B-FP8/20260915T161838Z-ba2694ea
LOG=$MY/autoprobe.log
OPT_PID=433163
PORT=8891
# The enablement lane's own validated stack: worktree + vcache of the specialist
# that reached "Application startup complete" with the stacked patches (001+002).
SPECIALIST=$SES/runs/specialist/8628519056b24176b3fa5c9a0e106dd6
WT=$SPECIALIST/wt

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
dc() { docker exec hyperloom-srv bash -lc "$*"; }

log "=== autoprobe armed (run pid $OPT_PID, port $PORT) ==="

# ---------------------------------------------------------------- 1. wait for close
# NB: liveness must NOT use `kill -0` — the optimizer runs as root inside the
# container, so an unprivileged kill -0 fails with EPERM and would look like
# "process gone" (that bug fired once at 18:40:56Z). Use ps -p instead.
DEADLINE=$(dc "/opt/envs/vllm/bin/python -c \"import json;print(int(json.load(open('$SES/state.json'))['deadline_unix']))\"" 2>/dev/null | tr -d '\r')
log "run deadline_unix=${DEADLINE:-unknown}"
for i in $(seq 1 300); do
  sr=$(dc "/opt/envs/vllm/bin/python -c \"import json;print(json.load(open('$SES/state.json')).get('stop_reason') or '')\"" 2>/dev/null | tr -d '\r')
  alive=no; ps -p "$OPT_PID" >/dev/null 2>&1 && alive=yes
  now=$(date +%s)
  if [ -n "$sr" ]; then log "run closed: stop_reason=$sr (pid alive=$alive)"; break; fi
  if [ "$alive" = no ]; then log "run process gone (stop_reason empty)"; break; fi
  if [ -n "${DEADLINE:-}" ] && [ "$now" -gt $((DEADLINE + 420)) ]; then
    log "deadline + 7min grace passed while pid still alive; proceeding"; break
  fi
  [ $((i % 10)) -eq 0 ] && log "waiting for close (poll $i, pid alive, $( [ -n "${DEADLINE:-}" ] && echo "$(( (DEADLINE - now) / 60 )) min to deadline" ))"
  sleep 30
done

# ---------------------------------------------------------------- 2. free the GPUs
sleep 20
holders=$(dc "ps -eo pid,args | grep -E 'vllm serve|VLLM::EngineCore' | grep -v grep | awk '{print \$1}'" 2>/dev/null | tr -d '\r')
if [ -n "$holders" ]; then
  log "leftover GPU holders: $(echo $holders | tr '\n' ' ')"
  for p in $holders; do dc "kill -TERM $p" >/dev/null 2>&1; done
  sleep 10
  for p in $holders; do dc "kill -KILL $p" >/dev/null 2>&1; done
fi
sleep 5
log "VRAM after cleanup: $(rocm-smi --showmemuse 2>/dev/null | grep 'VRAM%' | head -2 | tr '\n' ' ')"

# ---------------------------------------------------------------- 3. launch
# Deliberately leave VLLM_CACHE_ROOT / TORCHINDUCTOR_CACHE_DIR at their container
# defaults: the enablement lane's validated boot (18:25-18:37, patch 002 recomputes
# the AOT key) left the correct, non-stale artifacts in /root/.cache/vllm, so a
# cold cache root here would only add ~3-5 min of recompilation.
log "default compile cache: $(dc "du -sh /root/.cache/vllm 2>/dev/null | cut -f1" 2>/dev/null | tr -d '\r')"

cat > "$MY/launch3.sh" <<EOF
#!/bin/bash
set -u
echo \$\$ > $MY/server.pid
echo "pid=\$\$ pgid=\$(ps -o pgid= -p \$\$ | tr -d ' ') started=\$(date -u +%FT%TZ)" >> $LOG
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$WT"
export VLLM_ROCM_USE_AITER=0
export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_LOGGING_LEVEL=INFO
export PYTHONDONTWRITEBYTECODE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
cd "$MY"
exec /opt/envs/vllm/bin/python /usr/local/bin/vllm serve \\
  /mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-FP8 \\
  --port $PORT --tensor-parallel-size=8 --gpu-memory-utilization 0.95 \\
  --max-model-len 6144 --trust-remote-code
EOF
chmod 755 "$MY/launch3.sh"
rm -f "$MY/server.log"
log "launching patched server (worktree=$WT)"
dc "setsid bash $MY/launch3.sh > $MY/server.log 2>&1 < /dev/null &" >/dev/null 2>&1

# ---------------------------------------------------------------- 4. wait for ready
ready=no
for i in $(seq 1 80); do
  code=$(curl -s -m 4 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
  if [ "$code" = "200" ]; then ready=yes; log "server ready after $((i * 20))s"; break; fi
  [ $((i % 6)) -eq 0 ] && log "boot poll $i: http=$code | $(dc "tail -1 $MY/server.log" 2>/dev/null | tr -d '\r' | cut -c1-140)"
  sleep 20
done

if [ "$ready" != yes ]; then
  log "ABORT: server never became ready; log tail follows"
  dc "tail -25 $MY/server.log" >> "$LOG" 2>&1
  dc "kill -TERM -\$(cat $MY/server.pid)" >/dev/null 2>&1
  exit 1
fi

# ---------------------------------------------------------------- 5. probe
log "running probe.py (warmup + single x3 + conc 8/32)"
dc "cd $MY && /opt/envs/vllm/bin/python $MY/probe.py --base-url http://127.0.0.1:$PORT --out $MY/probe_results.json --max-tokens 256" >> "$LOG" 2>&1
rc=$?
log "probe exit=$rc"

# ---------------------------------------------------------------- 6. stop + summarize
pgid=$(dc "cat $MY/server.pid" 2>/dev/null | tr -d '\r')
[ -n "$pgid" ] && dc "kill -TERM -$pgid" >/dev/null 2>&1
sleep 12
dc "kill -KILL -$pgid" >/dev/null 2>&1
log "server stopped (pgid=$pgid); VRAM now: $(rocm-smi --showmemuse 2>/dev/null | grep 'VRAM%' | head -2 | tr '\n' ' ')"

if [ -f "$MY/probe_results.json" ]; then
  python3 - "$MY/probe_results.json" "$MY/PROBE_SUMMARY.md" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
s = r["single_stream_median"]
lines = [
    "# FP8->BF16 dequant-emulation probe (MI250X, TP8, vLLM)",
    "",
    f"- measured_at: {r['measured_at']}",
    f"- model: {r['model']}",
    f"- prompt tokens: {r['prompt_tokens']}",
    "",
    "## Single stream (median of 3)",
    f"- TTFT: {s['ttft_ms']:.0f} ms",
    f"- decode: {s['decode_tps']:.1f} tok/s",
    f"- TPOT: {s['tpot_ms']:.0f} ms",
    "",
    "## Concurrency",
]
for conc, c in r["concurrency"].items():
    lines += [
        f"- conc {conc} (osl {c['osl_target']}): aggregate {c['aggregate_output_tps']:.1f} output tok/s, "
        f"mean TTFT {c['mean_ttft_ms']:.0f} ms, mean TPOT {c['mean_tpot_ms']:.0f} ms, wall {c['wall_s']:.1f} s",
    ]
lines += ["", "Reference on the same host: llama.cpp Q8_0 arm = 47.28 tok/s (decode).", ""]
open(sys.argv[2], "w").write("\n".join(lines))
print("\n".join(lines))
PY
  log "summary written"
else
  log "no probe_results.json produced"
fi
log "=== autoprobe done ==="
