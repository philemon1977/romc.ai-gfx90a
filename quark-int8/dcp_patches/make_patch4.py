import difflib, sys
rel = "v1/attention/backends/mla/rocm_aiter_mla_sparse.py"
live = "/home/qiba/ai/recipes/patches/gfx90a/ct_w4a16_dsv41_n0918/tree/" + rel
old = open("base/backend.pre0004.py", encoding="utf8").read()
new = open(live, encoding="utf8").read()
if old == new:
    sys.exit("no diff: 0004 未生效？")
d = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                 fromfile="a/" + rel, tofile="b/" + rel, n=3))
hdr = "# DCP-D: metadata exposes local-shard seq_lens (needed by layer dcp_manager.combine);"
hdr += chr(10) + "#        row length min(ctx, topk) now uses the LOCAL shard length."
hdr += chr(10) + "# REQUIRES 0001+0003 applied first."
hdr += chr(10) + "# APPLY: cd /home/qiba/ai/recipes/patches/gfx90a/ct_w4a16_dsv41_n0918/tree && patch -p1 < PATCH"
hdr += chr(10)
open("0004_gfx90a_dcp_local_seq_lens.patch", "w", encoding="utf8").write(hdr + d)
print("hunks=%d lines=%d" % (sum(1 for l in d.splitlines() if l.startswith("@@ ")), len(d.splitlines())))
