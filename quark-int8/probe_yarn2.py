V = "/usr/local/lib/python3.12/dist-packages/vllm"
p = f"{V}/model_executor/layers/rotary_embedding/__init__.py"
src = open(p).read().splitlines()
hits = [i for i, l in enumerate(src) if "scaling_type" in l]
print(f"scaling_type 出现 {len(hits)} 次")
for i in hits:
    print(f"{i+1}: {src[i].strip()[:130]}")
