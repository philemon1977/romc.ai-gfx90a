import re
V = "/usr/local/lib/python3.12/dist-packages/vllm"
p = f"{V}/model_executor/layers/rotary_embedding/__init__.py"
src = open(p).read().splitlines()
start = None
for i, l in enumerate(src):
    if re.match(r"\s*def get_rope\(", l):
        start = i
        break
if start is not None:
    for j in range(start, min(start + 95, len(src))):
        print(f"{j+1}: {src[j][:150]}")
else:
    print("get_rope not found")
