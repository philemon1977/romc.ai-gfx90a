#!/usr/bin/env bash
# Wait until some MI250X GCD has >= 25 GB free HBM (so a vLLM engine can start).
for i in $(seq 1 480); do
  free_gb=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | \
    awk -F, '{printf "%.0f\n", ($2-$3)/1e9}' | sort -rn | head -1)
  if [ -n "$free_gb" ] && [ "$free_gb" -ge 25 ]; then
    echo "GPU_FREE: max free HBM = ${free_gb} GB after $((i-1)) minutes"; exit 0
  fi
  sleep 60
done
echo "GPU_STILL_BUSY after 8h"; exit 1
