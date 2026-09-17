P=/usr/local/lib/python3.12/dist-packages
echo "===== qwen3_5 special handling inside quark ====="
grep -rln "qwen3_5\|Qwen3_5\|Ornith" $P/quark | head
echo "===== ModelQuantizer.quantize_model signature & file2file branch ====="
sed -n '133,175p' $P/quark/torch/quantization/api.py
sed -n '290,330p' $P/quark/torch/quantization/api.py
echo "===== exclude matching body (243+) ====="
sed -n '320,384p' $P/quark/torch/quantization/file2file_quantization.py
echo "===== QConfig exclude doc/semantics ====="
sed -n "$(grep -n 'class QConfig' $P/quark/torch/quantization/config/config.py | cut -d: -f1),+45p" $P/quark/torch/quantization/config/config.py | grep -A6 "exclude"
