P=/usr/local/lib/python3.12/dist-packages
echo "===== template.py qwen3_5 ====="
grep -n "qwen3_5\|def \|class " $P/quark/torch/quantization/config/template.py | head -40
echo "===== model_preparation.py qwen3_5 ====="
grep -n "qwen3_5" $P/quark/torch/utils/llm/model_preparation.py | head
echo "===== _apply_weight_converters ====="
sed -n '115,173p' $P/quark/torch/quantization/file2file_quantization.py
echo "===== exclude match helper ====="
grep -rn "def is_layer_matched\|def match_pattern\|re.match\|fullmatch\|fnmatch" $P/quark/torch/quantization/config/config.py $P/quark/common/*.py $P/quark/common/**/*.py 2>/dev/null | head -10
