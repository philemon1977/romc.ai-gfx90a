#!/usr/bin/env bash
# dedup_undo.sh -- break hardlink sharing for the files listed in PLAN, giving $INTO
# private inodes again (i.e. restore the pre-dedup state, doubling disk use for those files).
# Only needed if you plan to *modify* one of the checkpoints in place.
#
# usage: sudo ./dedup_undo.sh --plan PLAN --into DUP_DIR [--apply]
set -uo pipefail
PLAN="" INTO="" APPLY=0
while [ $# -gt 0 ]; do case "$1" in
  --plan) PLAN=$2; shift 2;; --into) INTO=$2; shift 2;; --apply) APPLY=1; shift;;
  *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$PLAN" ] && [ -n "$INTO" ] || { echo "usage: sudo $0 --plan F --into DIR [--apply]" >&2; exit 2; }

df_avail=$(df -B1 --output=avail "$(dirname "$INTO")" | tail -1 | tr -d ' ')
need=$(awk -F'\t' '{s+=$2} END{print s}' "$PLAN")
echo "plan bytes: $(numfmt --to=iec "$need")   fs avail: $(numfmt --to=iec "$df_avail")"
[ "$need" -lt "$df_avail" ] || { echo "FATAL: not enough free space to un-share"; exit 1; }

while IFS=$'\t' read -r rel sz; do
  [ -n "$rel" ] || continue
  dst="$INTO/$rel"
  [ -f "$dst" ] || continue
  h=$(stat -c %h "$dst")
  if [ "$h" = 1 ]; then printf '  private  %s\n' "$rel"; continue; fi
  printf '  un-share %-42s links=%s\n' "$rel" "$h"
  [ "$APPLY" = 1 ] || continue
  chmod u+w "$dst" 2>/dev/null || true
  cp -a --sparse=always "$dst" "$dst.undup" || exit 1   # cp truncates the copy -> new inode
  mv -T "$dst.undup" "$dst" || exit 1
done < "$PLAN"
[ "$APPLY" = 1 ] || echo "DRY RUN - re-run with --apply"
