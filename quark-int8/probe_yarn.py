V = "/usr/local/lib/python3.12/dist-packages/vllm"
src = open(f"{V}/model_executor/layers/rotary_embedding/__init__.py").read().splitlines()
for i, l in enumerate(src):
    if 'scaling_type == "yarn"' in l or 'scaling_type in' in l or 'YaRN' in l:
        print(f"--- line {i+1} ---")
        for j in range(i, min(i + 30, len(src))):
            print(f"{j+1}: {src[j][:150]}")
        break
