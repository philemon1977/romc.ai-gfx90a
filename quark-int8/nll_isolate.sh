#!/usr/bin/env bash
# 两臂隔离：① SPEC=0（当前配方其余不变）② SPEC=5 + 前缀缓存开（撤掉我改的那一项）
set -uo pipefail
L_NOPFX=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int4w4a16ct_vllm_rocm72_mtp1_256k_8116_ornith_mi250dx8.sh
L_PFX=/tmp/launcher_pfxcache_on.sh
REPO=/home/qiba/ROCm.AI/quark-int8
exec > >(tee -a "$REPO/logs/nll_isolate.log") 2>&1
run_arm () {  # tag launcher spec
  local tag=$1 L=$2 spec=$3
  local pidf=/home/qiba/ai/logs/ornith397b-8116.pid
  [ -f "$pidf" ] && { local p; p=$(cat "$pidf" 2>/dev/null || true); [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null; rm -f "$pidf"; }
  for _ in $(seq 1 120); do nc -z 127.0.0.1 8116 2>/dev/null && { sleep 15; continue; }
    free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    [ "${free:-0}" -ge 62 ] && break; sleep 10; done
  export PORT=8116 SPEC="$spec"
  echo; echo "=== ARM $tag : SPEC=$spec  launcher=$(basename $L) $(date -u +%H:%M:%S) ==="
  bash "$L" >/dev/null || { echo "LAUNCH FAILED"; return 1; }
  for i in $(seq 1 80); do curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && break; sleep 10; done
  if ! curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1; then echo "未就绪"; return 1; fi
  python3 -u "$REPO/diag_nll.py" 8116 ornith "$tag" 2>&1 | grep -aE "NLL=|前 5"
  local p2; p2=$(cat "$pidf" 2>/dev/null || true); [ -n "${p2:-}" ] && kill -TERM -"$p2" 2>/dev/null
  sleep 20
}
run_arm SPEC3-noPfxCache "$L_NOPFX" 3
VLLM_EXTRA_ARGS="--max-num-batched-tokens 4096" run_arm SPEC5-MaxTok4096 "$L_NOPFX" 5
echo "=== nll_isolate done $(date -u +%FT%TZ) ==="
