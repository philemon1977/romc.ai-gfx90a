import re
p = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/quant_utils.py"
src = open(p).read().splitlines()
for i, l in enumerate(src):
    if re.match(r"\s*def should_ignore_layer", l) or re.match(r"\s*def _match_name", l) or "def _do_ignore" in l:
        for j in range(i, min(i + 45, len(src))):
            print(f"{j+1}: {src[j][:150]}")
        print("-----")
