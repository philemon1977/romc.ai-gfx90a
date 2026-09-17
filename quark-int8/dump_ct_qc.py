import json
D = "/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-CT-Int4-W4A16"
qc = json.load(open(f"{D}/config.json"))["quantization_config"]
print(json.dumps(qc, ensure_ascii=False))
