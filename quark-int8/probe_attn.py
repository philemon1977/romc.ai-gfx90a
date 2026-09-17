import re
V = "/usr/local/lib/python3.12/dist-packages/vllm"
print("=== ROCm 平台支持的 attention backends ===")
src = open(f"{V}/platforms/rocm.py").read()
m = re.search(r"def get_supported_attention_backends.*?(?=\n    def |\nclass )", src, re.S)
print((m.group(0)[:1800] if m else "not found"))
print("\n=== KV layout 相关开关 ===")
for pat in ("kv_cache_layout", "LBHNC", "NHD", "HND", "VLLM_ROCM_USE_AITER_PAGED_ATTN"):
    hits = []
    for f in ("_aiter_ops.py", "platforms/rocm.py", "envs.py", "v1/attention/backends/rocm_attn.py"):
        try:
            for i, l in enumerate(open(f"{V}/{f}").read().splitlines()):
                if pat in l:
                    hits.append(f"{f}:{i+1}: {l.strip()[:110]}")
        except FileNotFoundError:
            pass
    print(f"--- {pat} ({len(hits)} hits)")
    for h in hits[:5]:
        print("   ", h)
