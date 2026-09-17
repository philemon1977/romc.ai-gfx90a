#!/usr/bin/env bash
# Wait for the 640K server, then run needle-in-a-haystack at several lengths.
# Output is tee'd to logs/c_needle.log so it can be watched from the shell.
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
IMG=vllm/vllm-openai-rocm:nightly
LOG="$REPO/logs/c_needle.log"
mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== c_run start $(date -u +%FT%TZ) ==="
"$REPO/watch_container.sh" ornith-c640k 900 8100 || exit 1
docker logs ornith-c640k 2>&1 | grep -oE "GPU KV cache size: [0-9,]+ tokens.*|Available KV cache memory: [0-9.]+ GiB" | tail -2
for T in 65536 262144 458752 540000; do
  echo "=== needle @ target $T (~$((T*114/100)) actual tokens) $(date -u +%H:%M:%S) ==="
  docker run --rm --entrypoint bash --network host -v "$REPO:/work" "$IMG" \
    -c "python3 -u /work/longctx_test.py 8100 Ornith-640k $T 512" 2>&1 | tail -4
done
echo "=== c_run done $(date -u +%FT%TZ) ==="
