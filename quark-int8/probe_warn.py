V = "/usr/local/lib/python3.12/dist-packages/vllm"
p = f"{V}/v1/core/kv_cache_utils.py"
src = open(p).read().splitlines()
for i, l in enumerate(src):
    if "no KV cache group could be identified" in l:
        for j in range(max(0, i - 45), min(i + 12, len(src))):
            print(f"{j+1}: {src[j][:150]}")
        break
