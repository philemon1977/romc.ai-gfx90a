import re
V = "/usr/local/lib/python3.12/dist-packages/vllm"
lines = open(f"{V}/envs.py").read().splitlines()
print("=== envs.py 里 block/page size 与 layout 相关 ===")
for i, l in enumerate(lines):
    if re.search(r"block_size|page_size|KV_CACHE_LAYOUT|kv_cache_layout|mamba", l, re.I) and re.search(r'"VLLM|Literal|: |=', l):
        print(f"{i+1}: {l.strip()[:150]}")
print("\n=== 240-260 行上下文（layout Literal）===")
for j in range(238, 258):
    print(f"{j+1}: {lines[j][:140]}")
