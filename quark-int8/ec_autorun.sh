#!/usr/bin/env bash
# Adaptive launcher for the expert-cache V2 validation run:
#   - waits until all 8 GCDs have a usable amount free
#   - picks the largest gpu-memory-utilization that fits right now
#   - launches V2 (slot mode), then waits for health and reports
REPO=/home/qiba/ROCm.AI/quark-int8
FLOOR_GIB=44                     # weights(48.2)+MTP(1.54)-freed(2.9) ~= 46.8 -> need headroom
for i in $(seq 1 240); do
  minfree=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 \
            | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  if [ "${minfree:-0}" -ge "$FLOOR_GIB" ]; then
    util=$(python3 -c "print(round(min(0.95, (${minfree}-0.7)/63.98), 2))")
    echo "[$(date -u +%H:%M:%S)] min free ${minfree} GiB -> util=${util}; launching V2 (8%% offload, pool=8)"
    EC_MODE=slot EC_POOL=8 "$REPO/ec_probe.sh" ec8 8 1 "$util"
    sleep 20
    exec "$REPO/watch_container.sh" ornith-ec8 900 8100
  fi
  sleep 30
done
echo "gave up after 2h (min free ${minfree:-?} GiB)"
