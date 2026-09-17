P=/usr/local/lib/python3.12/dist-packages
echo "===== Int8PerChannelSpec / Int8PerTensorSpec ====="
sed -n '1002,1090p' $P/quark/torch/quantization/config/config.py
echo "===== QLayerConfig ctor ====="
sed -n '389,430p' $P/quark/torch/quantization/config/config.py
echo "===== template get QConfig (860-895) ====="
sed -n '860,895p' $P/quark/torch/quantization/config/template.py
echo "===== vLLM QuarkConfig from_config / get_name ====="
grep -n "def get_name\|def from_config\|def get_quant_config" $P/vllm/model_executor/layers/quantization/quark/quark.py
sed -n "$(grep -n 'def from_config' $P/vllm/model_executor/layers/quantization/quark/quark.py | head -1 | cut -d: -f1),+30p" $P/vllm/model_executor/layers/quantization/quark/quark.py
