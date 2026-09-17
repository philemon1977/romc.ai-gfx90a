V = "/usr/local/lib/python3.12/dist-packages/vllm"
src = open(f"{V}/model_executor/layers/rotary_embedding/__init__.py").read().splitlines()
for j in range(239, 280):
    print(f"{j+1}: {src[j][:150]}")
