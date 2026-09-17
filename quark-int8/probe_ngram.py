import re
V = "/usr/local/lib/python3.12/dist-packages/vllm"
p = f"{V}/config/speculative.py"
src = open(p).read()
print("=== 支持的 method 列表 ===")
m = re.search(r"MTPModelTypes\s*=\s*[^\n]+", src)
print(m.group(0)[:300] if m else "?")
for pat in (r'"ngram"', r"ngram", r"def use_eagle", r"def use_ngram", r"prompt_lookup"):
    hits = [i + 1 for i, l in enumerate(src.splitlines()) if re.search(pat, l)]
    print(f"{pat}: lines {hits[:8]}")
print("\n=== use_eagle 实现 ===")
lines = src.splitlines()
for i, l in enumerate(lines):
    if "def use_eagle" in l:
        for j in range(i, min(i + 12, len(lines))):
            print(f"{j+1}: {lines[j][:140]}")
        break
print("\n=== NGram 相关配置项 ===")
for i, l in enumerate(lines):
    if re.search(r"prompt_lookup|ngram_", l):
        print(f"{i+1}: {l.strip()[:140]}")
