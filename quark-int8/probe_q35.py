V = "/usr/local/lib/python3.12/dist-packages/vllm"
print("=== models/utils.py 355-390 ===")
for i, l in enumerate(open(f"{V}/model_executor/models/utils.py").read().splitlines()[354:390], start=355):
    print(f"{i}: {l[:150]}")
print("\n=== layers/linear.py 750-768 (weight_loader) ===")
for i, l in enumerate(open(f"{V}/model_executor/layers/linear.py").read().splitlines()[749:768], start=750):
    print(f"{i}: {l[:150]}")
print("\n=== qwen3_5.py 280-295 ===")
for i, l in enumerate(open(f"{V}/model_executor/models/qwen3_5.py").read().splitlines()[279:295], start=280):
    print(f"{i}: {l[:150]}")
