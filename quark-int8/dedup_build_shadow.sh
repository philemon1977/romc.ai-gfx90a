#!/usr/bin/env bash
# Build Ornith-1.5-397B-Quark-Int8-Attn-dedup: a drop-in replacement for the -Attn
# checkpoint whose byte-identical shards are the SAME inodes as the base Int8 checkpoint.
# No data is copied (hardlinks only), no existing file is modified -> needs no root.
set -uo pipefail

ROOT=/mnt/kioxia-cm6-3t8/ai/models/ornith-ai
V1="$ROOT/Ornith-1.5-397B-Quark-Int8"                 # base  (v1)
V2="$ROOT/Ornith-1.5-397B-Quark-Int8-Attn"            # attn  (v2)  -- stays untouched
NEW="$ROOT/Ornith-1.5-397B-Quark-Int8-Attn-dedup"     # output
HERE="$(cd "$(dirname "$0")" && pwd)"
IDT="$HERE/dedup_identical.txt"; DIF="$HERE/dedup_differ.txt"

[ -e "$NEW" ] && { echo "FATAL: $NEW already exists (refusing to touch)"; exit 1; }
[ -s "$IDT" ] && [ -s "$DIF" ] || { echo "FATAL: manifests missing, run dedup_verify.sh first"; exit 1; }

mkdir "$NEW" "$NEW/assets" || exit 1
created=0; failed=0

link() { # $1 = source absolute path, $2 = relative path
  local src=$1 rel=$2 dst="$NEW/$2"
  mkdir -p "$(dirname "$dst")"
  if ln "$src" "$dst"; then created=$((created+1)); else
    echo "  FAIL ln: $rel -> $src" >&2; failed=$((failed+1)); fi
}

# 1. files that are byte-identical in both -> take v1's inode
while IFS=$'\t' read -r rel sz; do [ -n "$rel" ] && link "$V1/$rel" "$rel"; done < "$IDT"
# 2. everything else in v2 (differing shards, config.json, index.json) -> v2's inode
while IFS=$'\t' read -r rel sz; do
  [ -n "$rel" ] || continue
  [ -e "$NEW/$rel" ] && continue          # already taken from v1 in step 1
  link "$V2/$rel" "$rel"
done < "$DIF"
# 3. dotfiles that ls skipped in the manifest pass
for f in "$V2"/.*; do
  b=${f##*/}; [ "$b" = "." ] || [ "$b" = ".." ] && continue
  [ -f "$f" ] || continue
  if cmp -s "$f" "$V1/$b"; then link "$V1/$b" "$b"; else link "$f" "$b"; fi
done

echo "linked $created entries, $failed failures"

# ---- verification -----------------------------------------------------------
echo "== file-set vs $V2"
( cd "$V2" && find . -type f | sort ) > /tmp/set_v2.txt
( cd "$NEW" && find . -type f | sort ) > /tmp/set_new.txt
diff /tmp/set_v2.txt /tmp/set_new.txt && echo "  IDENTICAL FILE SET"
echo "== every payload file must now have >=2 links"
one=$(cd "$NEW" && find . -type f -links 1 | sed 's|^\./||')
[ -z "$one" ] && echo "  OK: no single-link files" || echo "  WARN single-link: $one"
echo "== index/tensor audit"
python3 - "$NEW" "$V2" <<'PY'
import json,os,struct,sys
def audit(d,label):
    j=json.load(open(os.path.join(d,'model.safetensors.index.json')))
    wm=j['weight_map']; shards=sorted(set(wm.values()))
    miss=[f for f in shards if not os.path.exists(os.path.join(d,f))]
    ntensors=0; nb=0; sizes=[]
    for f in shards:
        p=os.path.join(d,f)
        with open(p,'rb') as fh:
            hl=struct.unpack('<Q',fh.read(8))[0]; h=json.loads(fh.read(hl))
        h.pop('__metadata__',None)
        assert '' not in h and all(v['data_offsets'][1]<=os.path.getsize(p)-8-hl for v in h.values()), f"offset overflow in {f}"
        ntensors+=len(h); sizes.append(len(h))
    tot_meta=os.path.getsize(os.path.join(d,'model.safetensors.index.json'))
    du=sum(os.path.getsize(os.path.join(d,f)) for f in shards)
    print(f"[{label}] shards={len(shards)} missing={len(miss)} tensors={ntensors} "
          f"bytes={du/1e9:.2f} GB config={json.load(open(os.path.join(d,'config.json')))['quantization_config']['quant_method']}")
    return ntensors,du
a=audit(sys.argv[1],'NEW -dedup'); b=audit(sys.argv[2],'ORIG -Attn')
print("match:", "PASS" if a==b else f"FAIL {a} vs {b}")
PY
echo "== spot byte-compare of 3 shared + 2 unique files against the originals"
for f in model-00002-of-00122.safetensors model-mtp.safetensors tokenizer.json; do
  cmp -s "$NEW/$f" "$V1/$f" && echo "  SAME-AS-V1 $f (inode $(stat -c %i "$NEW/$f"), links $(stat -c %h "$NEW/$f"))"
done
for f in model-00001-of-00122.safetensors config.json; do
  cmp -s "$NEW/$f" "$V2/$f" && echo "  SAME-AS-V2 $f (inode $(stat -c %i "$NEW/$f"), links $(stat -c %h "$NEW/$f"))"
done
echo
echo "real on-disk cost of the new dir (bytes not shared with v1):"
du -sh "$NEW" 2>/dev/null | cut -f1 | sed "s/^/  du -sh  /"
echo "  unique bytes (files whose inode belongs to v2 only):"
python3 - "$NEW" "$V1" <<'PY'
import os,sys
new,v1=sys.argv[1],sys.argv[2]
i1={os.stat(os.path.join(v1,f)).st_ino for f in os.listdir(v1) if os.path.isfile(os.path.join(v1,f))}
u=sum(os.path.getsize(os.path.join(dp,f)) for dp,_,fs in os.walk(new) for f in fs
      if os.stat(os.path.join(dp,f)).st_ino not in i1)
print(f"    {u/1e9:.2f} GB")
PY
