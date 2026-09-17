#!/usr/bin/env bash
# dedup_link.sh -- collapse byte-identical checkpoint files into hardlinks.
#
# Both checkpoints live on the SAME ext4 filesystem (/dev/nvme3n1p1), so hardlinks
# are legal and are strictly better than symlinks here:
#   * every reader (vLLM / safetensors mmap / python open) sees a normal file
#   * no dangling path if the model dir is bind-mounted alone into a container
#   * no second copy ever gets written
#
# Only files listed (byte-verified identical by cmp) in dedup_identical.txt are touched.
# Everything else stays exactly as it is, so BOTH dirs remain fully self-contained
# and usable: the model you point vLLM at needs no knowledge of the other one.
#
# usage:
#   sudo ./dedup_link.sh --plan PLAN --from BASE_DIR --into DUP_DIR [--apply] [--reverify] [--readonly]
#
#   --plan PATH     manifest produced by dedup_verify.sh (TSV: relpath<TAB>bytes)
#   --from DIR      the checkpoint that keeps its inodes (usually the base, v1)  [read-only source]
#   --into DIR      the checkpoint whose duplicate files become hardlinks        [modified]
#   --apply         actually mutate (default: dry-run)
#   --reverify      cmp each file again before linking (~2x the size of the plan in IO)
#   --readonly      chmod a-w the linked files so no tool can append/rewrite them in place
set -uo pipefail

PLAN="" FROM="" INTO="" APPLY=0 REVERIFY=0 READONLY=0
while [ $# -gt 0 ]; do case "$1" in
  --plan) PLAN=$2; shift 2;; --from) FROM=$2; shift 2;; --into) INTO=$2; shift 2;;
  --apply) APPLY=1; shift;; --reverify) REVERIFY=1; shift;; --readonly) READONLY=1; shift;;
  *) echo "unknown arg: $1" >&2; exit 2;;
esac; done
[ -n "$PLAN" ] && [ -n "$FROM" ] && [ -n "$INTO" ] || { sed -n '15,22p' "$0"; exit 2; }

# --- preflight -------------------------------------------------------------
echo "plan  : $PLAN"
echo "base  : $FROM   (inodes kept)"
echo "dup   : $INTO   (files replaced by hardlinks)"
[ -d "$FROM" ] && [ -d "$INTO" ] || { echo "FATAL: dir missing"; exit 1; }
[ "$APPLY" = 1 ] && { [ -w "$INTO" ] || echo "NOTE: $INTO not writable -> run with sudo"; }

d1=$(stat -c %d "$FROM"); d2=$(stat -c %d "$INTO")
[ "$d1" = "$d2" ] || { echo "FATAL: different filesystem (dev $d1 vs $d2) -> hardlinks impossible"; exit 1; }
echo "same filesystem: dev=$d1"
case "$(stat -f -c %T "$INTO")" in
  ext*|xfs|btrfs) : ;;
  *) echo "WARN: unexpected fstype $(stat -f -c %T "$INTO")";;
esac
command -v python3 >/dev/null || { echo "FATAL: python3 required for the index check"; exit 1; }

# refuse if $INTO already contains links pointing outside its own tree
if find "$INTO" -maxdepth 1 -type l | grep -q .; then
  echo "FATAL: $INTO contains symlinks; resolve them first (ln -s would double-share)"; exit 1
fi

# --- main loop -------------------------------------------------------------
linked=0; skipped=0; bytes=0; fails=0
while IFS=$'\t' read -r rel sz; do
  [ -n "$rel" ] || continue
  case "$rel" in */|.*.swp) continue;; esac
  src="$FROM/$rel"; dst="$INTO/$rel"
  [ -f "$src" ] || { echo "  skip (no base file): $rel"; skipped=$((skipped+1)); continue; }
  if [ ! -f "$dst" ]; then echo "  skip (missing in dup): $rel"; skipped=$((skipped+1)); continue; fi
  [ "$(stat -c %s "$src")" = "$(stat -c %s "$dst")" ] || { echo "  skip (size drift): $rel"; fails=$((fails+1)); continue; }
  if [ "$(stat -c %i "$src")" = "$(stat -c %i "$dst")" ]; then echo "  already linked: $rel"; skipped=$((skipped+1)); continue; fi
  if [ "$REVERIFY" = 1 ] && ! cmp -s "$src" "$dst"; then echo "  skip (NOT identical): $rel"; fails=$((fails+1)); continue; fi

  bytes=$((bytes + sz)); linked=$((linked+1))
  printf '  %-42s %8.2f GB  links=%s\n' "$rel" "$(echo "$sz/1e9" | bc -l)" "$(stat -c %h "$dst")"
  [ "$APPLY" = 1 ] || continue

  tmp="$INTO/.dedup.$$.tmp"
  ln "$src" "$tmp"                     || { echo "  FATAL: ln failed"; exit 1; }
  mv -T "$tmp" "$dst"                  || { echo "  FATAL: atomic rename failed"; exit 1; }
  # rename() replaced the dentry atomically: at no point was $dst missing/short
  [ "$READONLY" = 1 ] && chmod a-w "$dst"
done < "$PLAN"

echo "------------------------------------------------------------"
printf 'files to share : %d   (%.2f GB reclaimed once old inodes drop)\n' "$linked" "$(echo "$bytes/1e9" | bc -l)"
printf 'skipped/mismatch: %d / %d\n' "$skipped" "$fails"
[ "$APPLY" = 1 ] || echo "DRY RUN - nothing changed. Re-run with --apply."

# --- post-checks -----------------------------------------------------------
if [ "$APPLY" = 1 ]; then
  python3 - "$INTO" <<'PY'
import json,os,sys
d=sys.argv[1]
j=json.load(open(os.path.join(d,'model.safetensors.index.json')))
wm=j['weight_map']; miss=[f for f in set(wm.values()) if not os.path.exists(os.path.join(d,f))]
tot=os.path.getsize(os.path.join(d,'model.safetensors.index.json'))
n=len(wm)
print(f"[index] {n} tensors -> {len(set(wm.values()))} shards, missing files: {len(miss)} {miss[:3]}")
bad=[]
for f in sorted(set(wm.values()))[:4]+[k for k in wm if 'layers.0.' in k][:4]:
    p=os.path.join(d,f)
    with open(p,'rb') as fh:
        hl=int.from_bytes(fh.read(8),'little'); h=json.loads(fh.read(hl))
    h.pop('__metadata__',None); bad += [f"{f}:{k}" for k in h if k not in wm]
print("[index] header/name check on a sample:", "OK" if not bad else f"MISMATCH {bad[:5]}")
PY
  echo "[du] per-dir (hardlinks counted once per dir):"
  du -sh "$FROM" "$INTO" 2>/dev/null
  echo "[du] combined real usage:"
  du -shc "$FROM" "$INTO" 2>/dev/null | tail -1
fi
