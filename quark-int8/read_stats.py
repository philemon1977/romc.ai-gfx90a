import json, sys
p = sys.argv[1] if len(sys.argv) > 1 else "/work/expert_cache_stats.json"
d = json.load(open(p))
g = d["global"]
print("GLOBAL:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items()}))
for k, v in list(d.get("sample_layers", {}).items())[:3]:
    print(" layer", k, json.dumps({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}))
