V = "/usr/local/lib/python3.12/dist-packages/vllm"
src = open(f"{V}/v1/core/kv_cache_utils.py").read().splitlines()
for j in range(2160, 2245):
    print(f"{j+1}: {src[j][:155]}")
