#!/usr/bin/env bash
# Separate the three sources of single-stream TPS variation, on ONE server (no restart):
#   A) identical prompt x5      -> pure timing noise
#   B) salted prompts x5        -> + content variation (what we have been reporting)
#   C) three workload types     -> how much TPS depends on how predictable the text is
# The point: "12-24% spread" was never characterised; if B >> A, the spread is content,
# not measurement noise, and the honest metric is a workload-labelled average.
set -uo pipefail
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
REPO=/home/qiba/ROCm.AI/quark-int8
LOG="$REPO/logs/prompt_sweep.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== prompt_sweep start $(date -u +%FT%TZ) ==="

pidf=/home/qiba/ai/logs/ornith397b-8116.pid
if ! curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1; then
  [ -f "$pidf" ] && { p=$(cat "$pidf" 2>/dev/null || true); [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null; rm -f "$pidf"; }
  for _ in $(seq 1 180); do
    busy=$(docker ps --format '{{.Names}}' | grep -c '^ornith-' || true)
    nc -z 127.0.0.1 8116 2>/dev/null && { sleep 15; continue; }
    free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    [ "$busy" = "0" ] && [ "${free:-0}" -ge 62 ] && break
    sleep 15
  done
  export PORT=8116 SPEC=5
  echo "=== 起服 SPEC=5 $(date -u +%H:%M:%S) ==="
  bash "$LAUNCH" >/dev/null || { echo "LAUNCH FAILED"; exit 1; }
  for i in $(seq 1 200); do curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && { echo "✅ ready (~$((i*5))s)"; break; }; sleep 5; done
fi
curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 || { echo "❌ NOT READY"; exit 1; }

python3 - <<'PY'
import json, secrets, statistics, time, urllib.request

def run(prompt, reps, label):
    tps, acc = [], []
    for i in range(reps):
        body = json.dumps({'model': 'ornith', 'prompt': prompt, 'max_tokens': 256,
                           'temperature': 0, 'ignore_eos': True}).encode()
        req = urllib.request.Request('http://127.0.0.1:8116/v1/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
        t0 = time.time(); d = json.load(urllib.request.urlopen(req, timeout=1800)); dt = time.time() - t0
        u = d['usage']
        tps.append(u['completion_tokens'] / dt)
        print(f"  [{label}] rep{i+1}: {tps[-1]:.2f} tok/s", flush=True)
    med = statistics.median(tps)
    print(f"[{label}] MEDIAN {med:.2f} tok/s  min {min(tps):.2f} max {max(tps):.2f} "
          f"spread {100*(max(tps)-min(tps))/med:.1f}%")
    return med

BASE = ("Explain in detail why int4 quantization reduces memory bandwidth pressure "
        "during decoding, covering weights and KV cache.")

# A) identical prompt -> timing noise only (greedy decode should be deterministic)
run(BASE, 5, "A 同一 prompt")

# B) salted prompts -> timing + content variation (our historical practice)
tps = []
for i in range(5):
    salt = secrets.token_hex(4)
    body = json.dumps({'model': 'ornith', 'prompt': f"[salt {salt}] {BASE}", 'max_tokens': 256,
                       'temperature': 0, 'ignore_eos': True}).encode()
    req = urllib.request.Request('http://127.0.0.1:8116/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time(); d = json.load(urllib.request.urlopen(req, timeout=1800)); dt = time.time() - t0
    tps.append(d['usage']['completion_tokens'] / dt)
    print(f"  [B 加盐 prompt] rep{i+1}: {tps[-1]:.2f} tok/s", flush=True)
med = statistics.median(tps)
print(f"[B 加盐 prompt] MEDIAN {med:.2f} tok/s  min {min(tps):.2f} max {max(tps):.2f} "
      f"spread {100*(max(tps)-min(tps))/med:.1f}%")

# C) workload types -> dependence on text predictability
run("Count slowly from one to ninety, writing each number in words on its own line.", 3, "C1 数数（高可预测）")
run("Repeat the following text exactly, without any changes: " + BASE, 3, "C2 复述（极高可预测）")
run("Write a short poem about the sea, then translate it into Chinese.", 3, "C3 创作（低可预测）")
PY
curl -s localhost:8116/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{" | sed 's/^/  /'
echo "=== prompt_sweep done $(date -u +%FT%TZ) ==="
