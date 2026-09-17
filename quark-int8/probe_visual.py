import json, os, struct, collections
D = "/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-CT-Int4-W4A16"
idx = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
vis = [k for k in idx if k.startswith("visual.") or k.startswith("model.visual.")]
print("visual tensors:", len(vis))
suf = collections.Counter(k.rsplit(".", 1)[-1] for k in vis)
print("suffixes:", suf.most_common(6))
for k in vis[:6]:
    f = idx[k]; p = os.path.join(D, f)
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
    m = hdr[k]; print(f"  {k:<45} {m['dtype']:<5} {m['shape']}")
# how about merger specifically
mg = [k for k in idx if "merger" in k]
print("merger tensors:", mg[:6])
