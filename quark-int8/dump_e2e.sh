#!/usr/bin/env bash
set -uo pipefail
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
REPO=/home/qiba/ROCm.AI/quark-int8
exec > >(tee -a "$REPO/logs/dump_e2e.log") 2>&1
pidf=/home/qiba/ai/logs/ornith397b-8116.pid
for _ in $(seq 1 120); do nc -z 127.0.0.1 8116 2>/dev/null && { sleep 15; continue; }
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "${free:-0}" -ge 62 ] && break; sleep 10; done
export PORT=8116 SPEC=5 PYTHONPATH="$REPO/moe_gemv_patch" MI250_MOE_GEMV=1 MI250_MOE_GEMV_DEBUG=1
bash "$LAUNCH" >/dev/null || exit 1
for i in $(seq 1 60); do curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && break; sleep 10; done
curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 || { echo "未就绪"; exit 1; }
python3 - <<'PY'
import json,urllib.request
body=json.dumps({'model':'ornith','prompt':'Count to three.','max_tokens':4,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8116/v1/completions',data=body,headers={'Content-Type':'application/json'})
print("请求OK:", json.load(urllib.request.urlopen(req,timeout=600))['usage'])
PY
SRV=$(ls -t /home/qiba/ai/logs/ornith397b/server-8116-*.log | head -1)
echo "=== DUMP ==="; grep -a "DUMP" "$SRV" | head -2
p2=$(cat "$pidf" 2>/dev/null || true); [ -n "${p2:-}" ] && kill -TERM -"$p2" 2>/dev/null
echo "done"
