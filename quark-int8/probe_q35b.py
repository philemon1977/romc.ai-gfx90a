V = "/usr/local/lib/python3.12/dist-packages/vllm"
p = f"{V}/model_executor/models/qwen3_5.py"
lines = open(p).read().splitlines()
print("file lines:", len(lines))
print("=== 285-300 (traceback 里的 289) ===")
for i in range(283, min(302, len(lines))):
    print(f"{i+1}: {lines[i][:150]}")
print("\n=== 类与权重映射 ===")
for i, l in enumerate(lines):
    if l.startswith("class ") or "hf_to_vllm_mapper" in l or "WeightsMapper" in l or "def load_weights" in l:
        print(f"{i+1}: {l[:140]}")
