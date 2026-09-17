import json, os, struct, collections
D = "/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-CT-Int4-W4A16"
cfg = json.load(open(os.path.join(D, "config.json")))
qc = cfg.get("quantization_config", {})
print("quant_method:", qc.get("quant_method"), "| format:", qc.get("format"))
print("config_groups:", json.dumps(qc.get("config_groups", {}))[:400])
ig = qc.get("ignore", [])
print(f"ignore entries: {len(ig)}; sample: {ig[:6]}")
idx = json.load(open(os.path.join(D, "model.safetensors.index.json")))["weight_map"]
print("tensors:", len(idx), "files:", len(set(idx.values())))
kinds = collections.Counter()
for k in idx:
    n = k.split(".")[-1]
    kinds[n] += 1
print("top suffixes:", kinds.most_common(8))
# inspect a few tensor headers
def show(name):
    f = idx[name]; p = os.path.join(D, f)
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
    m = hdr[name]; print(f"  {name.split('layers.')[-1][:60]:<60} {m['dtype']:<6} {m['shape']}")
for key in [k for k in idx if "experts.0." in k][:4]:
    show(key)
for key in [k for k in idx if "self_attn.q_proj" in k][:2]:
    show(key)
for key in [k for k in idx if k.startswith("mtp.")][:2]:
    show(key)
