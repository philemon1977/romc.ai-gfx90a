P=/usr/local/lib/python3.12/dist-packages
sed -n '1240,1420p' $P/quark/torch/quantization/config/template.py
echo "===== callers of this template function ====="
grep -rn "$(sed -n '1240,1300p' $P/quark/torch/quantization/config/template.py | grep -oP 'def \K\w+' | head -1)" $P/quark --include=*.py | grep -v __pycache__ | head
