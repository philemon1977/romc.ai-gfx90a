#!/usr/bin/env bash
# 起服（当前配方，补丁关）并跑两条 NLL 路径的对照
set -uo pipefail
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
REPO=/home/qiba/ROCm.AI/quark-int8
exec > >(tee -a "$REPO/logs/nll_diag.log") 2>&1
pidf=/home/qiba/ai/logs/ornith397b-8116.pid
[ -f "$pidf" ] && { p=$(cat "$pidf" 2>/dev/null || true); [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null; rm -f "$pidf"; }
for _ in $(seq 1 120); do nc -z 127.0.0.1 8116 2>/dev/null && { sleep 15; continue; }
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "${free:-0}" -ge 62 ] && break; sleep 10; done
export PORT=8116 SPEC=5
echo "=== 起服 SPEC=5（当前配方）$(date -u +%H:%M:%S) ==="
bash "$LAUNCH" >/dev/null || exit 1
for i in $(seq 1 80); do curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && break; sleep 10; done
curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 || { echo "未就绪"; exit 1; }
python3 -u "$REPO/diag_nll.py" 8116 ornith "SPEC5-noPfxCache"
p2=$(cat "$pidf" 2>/dev/null || true); [ -n "${p2:-}" ] && kill -TERM -"$p2" 2>/dev/null
echo "=== done $(date -u +%FT%TZ) ==="
