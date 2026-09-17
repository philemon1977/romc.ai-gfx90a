import re
V = "/usr/local/lib/python3.12/dist-packages/vllm"
print("=== qwen3_next rope 取参 ===")
src = open(f"{V}/model_executor/models/qwen3_next.py").read().splitlines()
for i, l in enumerate(src):
    if re.search(r"rope_scaling|rope_parameters|max_position|get_rope", l):
        print(f"{i+1}: {l.strip()[:150]}")
print("\n=== CLI 是否有 --rope-scaling / --hf-overrides ===")
args = open(f"{V}/engine/arg_utils.py").read()
for pat in (r"--rope-scaling", r"rope_scaling", r"hf_overrides", r"max_model_len"):
    print(f"{pat}: {'yes' if re.search(pat, args) else 'no'}")
print("\n=== mrope 的 yarn 分支 ===")
m = open(f"{V}/model_executor/layers/rotary_embedding/mrope.py").read().splitlines()
for i, l in enumerate(m):
    if re.search(r"yarn|YaRN|scaling_factor|original_max_position", l):
        print(f"{i+1}: {l.strip()[:140]}")
