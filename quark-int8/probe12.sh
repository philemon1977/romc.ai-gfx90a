P=/usr/local/lib/python3.12/dist-packages
echo "===== api.py file2file wiring ====="
grep -n "file2file\|quantize_model_per_safetensor\|adaptive\|class .*Quantizer\|def quantize" $P/quark/torch/quantization/api.py | head -20
echo "===== SplitFusedExperts ====="
sed -n '105,172p' $P/quark/torch/quantization/weight_convert.py
echo "===== WeightConverter ctor ====="
sed -n '408,470p' $P/quark/torch/quantization/weight_convert.py
echo "===== quark exclude matching ====="
sed -n '243,320p' $P/quark/torch/quantization/file2file_quantization.py
echo "===== vLLM QuarkConfig.from_config keys ====="
grep -n "global_quant_config\|quant_method\|from_config\|def get_from_config" $P/vllm/model_executor/layers/quantization/quark/quark.py | head
echo "===== vLLM should_ignore pattern semantics ====="
grep -n "def should_ignore_layer\|def _match_name\|re:\|fnmatch" $P/vllm/model_executor/layers/quantization/utils/quant_utils.py | head -15
echo "===== bundled example jsons ====="
find $P/quark -name "*.json" | grep -viE "test|doc" | head
