V = "/usr/local/lib/python3.12/dist-packages/vllm"
lines = open(f"{V}/config/speculative.py").read().splitlines()
print("=== num_speculative_tokens 默认 ===")
for i, l in enumerate(lines):
    if "num_speculative_tokens" in l and ("Field" in l or "default" in l):
        print(f"{i+1}: {l.strip()[:140]}")
print("=== ngram 分支 (1170-1210) ===")
for j in range(1170, 1212):
    print(f"{j+1}: {lines[j][:145]}")
