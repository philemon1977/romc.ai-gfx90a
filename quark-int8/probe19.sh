P=/usr/local/lib/python3.12/dist-packages
echo "===== template qconfig method name ====="
awk 'NR>=800 && NR<=830 && /def /' $P/quark/torch/quantization/config/template.py
grep -n "def create_quant_config\|def get_qconfig\|def create_qconfig\|def to_qconfig\|def get_quant_config" $P/quark/torch/quantization/config/template.py
echo "===== NaiveDynamic examples in wheel ====="
grep -rln "naive_dynamic\|NaiveDynamic" $P/quark 2>/dev/null | grep -v __pycache__ | head -3
echo "===== vLLM uses export.weight_format? ====="
grep -rn "weight_format\|real_quantized\|pack_method" $P/vllm/model_executor/layers/quantization/quark/*.py | head -5
echo "===== direct_quantize_checkpoint full sig ====="
sed -n "$(grep -n 'def direct_quantize_checkpoint' $P/quark/torch/quantization/api.py | cut -d: -f1),+28p" $P/quark/torch/quantization/api.py
echo "===== QConfig ctor params ====="
sed -n '136,190p' $P/quark/torch/quantization/config/config.py | grep -E "param|: |def " | head -20
