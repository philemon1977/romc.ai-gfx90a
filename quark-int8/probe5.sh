P=/usr/local/lib/python3.12/dist-packages
V=$P/vllm
echo "=== quark CLI? ==="
ls /usr/local/bin /usr/bin 2>/dev/null | grep -i quark || echo "no quark executable (library-only)"
python3 -c "import quark.scripts" 2>&1|tail -1
echo "=== quark quantizer classes ==="
grep -rn "class .*Quantizer" $P/quark/torch/quantization/ --include=*.py | grep -v test | head -15
ls $P/quark/torch/quantization/ | head -30
echo "=== non-dynamic / offline ==="
find $P/quark -iname "*non_dynamic*" -o -iname "*offline*" | head
echo "=== _is_w8a8_int8 exact accepted config ==="
sed -n '505,585p' $V/model_executor/layers/quantization/quark/quark.py
echo "=== QuarkW8A8Int8MoEMethod create_weights ==="
sed -n '646,760p' $V/model_executor/layers/quantization/quark/quark_moe.py
