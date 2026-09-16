set -x
python3 -c "import vllm; print('vllm', vllm.__version__); print(vllm.__file__)"
python3 -c "import quark; print('quark', quark.__version__, quark.__file__)" 2>&1 | tail -1
which quark || echo "NO quark CLI"
ls /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/quark/ 2>/dev/null
echo "=== arch support ==="
grep -rn "qwen3_5\|Qwen3_5\|Ornith\|ornith" /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py /usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/config.py 2>/dev/null | head
echo "=== quark int8/moe in vllm ==="
grep -n "INT8\|int8\|Int8" /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/quark/*.py | grep -iv "fp8" | head -40
echo "=== quark schemes list ==="
grep -n "class Quark\|QUARK schemes\|__all__" /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/quark/*.py | head -20
