#!/usr/bin/env bash
# Strictly serial: CK int4 numeric verification (S1, short) then the replicated MTP sweep.
set -uo pipefail
cd /home/qiba/ROCm.AI/quark-int8
echo "=== chain2 start $(date -u +%FT%TZ) ==="
bash ./run_ck_int4_v2.sh
echo "=== CK 段结束，转入 MTP 复测 ==="
bash ./mtp_reps.sh
echo "=== chain2 done $(date -u +%FT%TZ) ==="
