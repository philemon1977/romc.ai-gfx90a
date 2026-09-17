#!/usr/bin/env bash
# Strictly serial: CK int4 numeric test first (short, needs the GCD briefly), then the
# deeper MTP sweep. Both wait for the GPU themselves; nothing else runs concurrently.
set -uo pipefail
cd /home/qiba/ROCm.AI/quark-int8
echo "=== chain start $(date -u +%FT%TZ) ==="
bash ./run_ck_int4.sh
echo "=== CK int4 段结束，转入 MTP 深扫 ==="
bash ./mtp_deep.sh
echo "=== chain done $(date -u +%FT%TZ) ==="
