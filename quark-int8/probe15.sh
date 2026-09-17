P=/usr/local/lib/python3.12/dist-packages
echo "===== Int8Scheme config ====="
sed -n '119,133p' $P/quark/torch/quantization/config/template.py
echo "===== model_preparation qwen3_5_moe (430-500) ====="
sed -n '430,500p' $P/quark/torch/utils/llm/model_preparation.py
echo "===== who uses SplitFusedExperts ====="
grep -rn "SplitFusedExperts" $P/quark $P/vllm 2>/dev/null | grep -v "def \|class \|__pycache__" | head
echo "===== file2file exclude loop ====="
sed -n '258,278p' $P/quark/torch/quantization/file2file_quantization.py
grep -n "exclude" $P/quark/torch/quantization/file2file_quantization.py | sed -n '1,20p'
echo "===== is_pattern_matched in config ====="
grep -rn "def .*match" $P/quark/torch/quantization/config/config.py | head
