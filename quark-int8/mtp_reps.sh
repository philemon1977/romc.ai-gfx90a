#!/usr/bin/env bash
# Re-measure MTP depth with REPETITIONS — the single-shot sweep produced a non-monotonic
# curve (n=5 64.95 > n=6 59.10, n=8 37.82 < n=10 46.56) whose step times are physically
# inconsistent (n=5 56.5 ms < n=4 62.9 ms), i.e. ~±10% run-to-run noise on one 256-token
# request. Same server, 5 requests each with a distinct prompt salt (project discipline:
# fresh salt per rep defeats prefix-cache hits), report the MEDIAN.
set -uo pipefail
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
REPO=/home/qiba/ROCm.AI/quark-int8
LOG="$REPO/logs/mtp_reps.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== mtp_reps start $(date -u +%FT%TZ) ==="

wait_free () {
  for _ in $(seq 1 180); do
    busy=$(docker ps --format '{{.Names}}' | grep -c '^ornith-' || true)
    nc -z 127.0.0.1 8116 2>/dev/null && { sleep 15; continue; }
    nc -z 127.0.0.1 8115 2>/dev/null && { sleep 15; continue; }
    free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    [ "$busy" = "0" ] && [ "${free:-0}" -ge 62 ] && return 0
    sleep 15
  done
  return 1
}

for n in 4 5 6; do
  pidf=/home/qiba/ai/logs/ornith397b-8116.pid
  [ -f "$pidf" ] && { p=$(cat "$pidf"); kill -TERM -"$p" 2>/dev/null; rm -f "$pidf"; }
  wait_free || { echo "[n=$n] GPU 不空"; continue; }
  export PORT=8116 SPEC="$n"
  echo "=== ARM SPEC=$n $(date -u +%H:%M:%S) ==="
  bash "$LAUNCH" >/dev/null || { echo "[n=$n] LAUNCH FAILED"; continue; }
  for i in $(seq 1 200); do curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && break; sleep 5; done
  curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 || { echo "[n=$n] ❌ NOT READY"; continue; }

  python3 - "$n" <<'PY'
import json, statistics, sys, time, urllib.request, secrets
n = sys.argv[1]
tps = []
for rep in range(5):
    salt = secrets.token_hex(4)
    prompt = (f"[salt {salt}] Explain in detail why int4 quantization reduces memory "
              f"bandwidth pressure during decoding, covering weights, activations and KV cache.")
    body = json.dumps({'model': 'ornith', 'prompt': prompt, 'max_tokens': 256,
                       'temperature': 0, 'ignore_eos': True}).encode()
    req = urllib.request.Request('http://127.0.0.1:8116/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time(); d = json.load(urllib.request.urlopen(req, timeout=1800)); dt = time.time() - t0
    u = d['usage']; r = u['completion_tokens'] / dt; tps.append(r)
    print(f"  [SPEC={n}] rep{rep+1}: {r:.2f} tok/s ({u['completion_tokens']} tok in {dt:.1f}s)")
print(f"[SPEC={n}] MEDIAN {statistics.median(tps):.2f} tok/s   min {min(tps):.2f}  max {max(tps):.2f}  spread {100*(max(tps)-min(tps))/statistics.median(tps):.1f}%")
PY
  curl -s localhost:8116/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{" | sed "s/^/  [SPEC=$n] /"
  p2=$(cat "$pidf" 2>/dev/null || true); [ -n "${p2:-}" ] && kill -TERM -"$p2" 2>/dev/null && echo "[n=$n] 已停服"
  sleep 25
done
echo "=== mtp_reps done $(date -u +%FT%TZ) ==="
