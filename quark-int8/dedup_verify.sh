#!/usr/bin/env bash
# Byte-exact verification of which files in the two Quark INT8 checkpoints are identical.
# Read-only. Produces dedup_identical.txt / dedup_differ.txt in the same directory.
set -uo pipefail

ROOT=/mnt/kioxia-cm6-3t8/ai/models/ornith-ai
A="$ROOT/Ornith-1.5-397B-Quark-Int8"          # v1  (experts-only INT8)
B="$ROOT/Ornith-1.5-397B-Quark-Int8-Attn"     # v2  (experts + attn INT8)
OUT="$(cd "$(dirname "$0")" && pwd)"
PAR=${PAR:-8}

same="$OUT/dedup_identical.txt"; diff_="$OUT/dedup_differ.txt"
: > "$same"; : > "$diff_"

check() {  # $1 = relative path
  local rel="$1" sz
  sz=$(stat -c %s "$B/$rel" 2>/dev/null) || { echo "MISSING $rel" >> "$diff_"; return; }
  if cmp -s "$A/$rel" "$B/$rel"; then
    echo -e "$rel\t$sz" >> "$same"
  else
    echo -e "$rel\t$sz" >> "$diff_"
  fi
}
export -f check; export A B same diff_

ls -1 "$A" | grep -v '^\.\+$' | while read -r f; do
  if [ -f "$A/$f" ]; then echo "$f"; fi
  if [ -d "$A/$f" ]; then (cd "$A" && find "$f" -type f); fi
done > "$OUT/dedup_allfiles.txt"

xargs -a "$OUT/dedup_allfiles.txt" -d '\n' -P "$PAR" -I{} bash -c 'check "$@"' _ {}

echo "================= SUMMARY ================="
awk -F'\t' '{n++; s+=$2} END {printf "identical : %3d files  %8.2f GB\n", n, s/1e9}' "$same"
awk -F'\t' '{n++; s+=$2} END {printf "differing : %3d files  %8.2f GB\n", n, s/1e9}' "$diff_"
echo "identical list:"; sort "$same" | awk -F'\t' '{printf "  %-40s %6.2f GB\n", $1, $2/1e9}' | head -80
