#!/usr/bin/env bash
# Serial: (1) prompt/workload sweep on one server, (2) single-variable knob ablations.
set -uo pipefail
cd /home/qiba/ROCm.AI/quark-int8
echo "=== chain3 start $(date -u +%FT%TZ) ==="
bash ./prompt_sweep.sh
echo "=== 转入 knob 消融 ==="
bash ./knob_ab.sh
echo "=== chain3 done $(date -u +%FT%TZ) ==="
