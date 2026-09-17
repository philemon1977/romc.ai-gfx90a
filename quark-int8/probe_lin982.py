V = "/usr/local/lib/python3.12/dist-packages/vllm"
lines = open(f"{V}/model_executor/layers/linear.py").read().splitlines()
print("=== linear.py 955-1000 (含 982) ===")
for i in range(954, min(1000, len(lines))):
    print(f"{i+1}: {lines[i][:150]}")
