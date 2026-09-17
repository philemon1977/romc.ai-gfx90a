"""Print what QuarkW8A8Int8MoEMethod.process_weights_after_loading leaves behind,
so the expert-cache V2 can shrink exactly those tensors."""
import re

p = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/quark/quark_moe.py"
src = open(p).read().splitlines()
start = None
found = False
for i, l in enumerate(src):
    if l.startswith("class QuarkW8A8Int8MoEMethod"):
        start = i
    if start and i > start and "def process_weights_after_loading" in l:
        for j in range(i, min(i + 75, len(src))):
            print(f"{j+1}: {src[j]}")
        found = True
        break
if not found:
    print("not found")

print("\n=== triton int8 kernel: weight/scale names ===")
k = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/triton_moe.py"
try:
    for i, l in enumerate(open(k).read().splitlines()):
        if re.search(r"w13_weight|w2_weight|weight_scale|def apply|def _apply", l):
            print(f"{i+1}: {l.strip()[:150]}")
except FileNotFoundError:
    print("triton_moe.py not found")
