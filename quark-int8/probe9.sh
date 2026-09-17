P=/usr/local/lib/python3.12/dist-packages
F=$P/quark/torch/quantization/file2file_quantization.py
echo "===== _is_linear_weight_tensor (matches fused MoE?) ====="
sed -n '191,258p' $F
echo "===== _export_quant_config ====="
sed -n '1803,1845p' $F
echo "===== QTensorConfig fields ====="
grep -n "class QTensorConfig\|class QLayerConfig\|class QConfig\|class QuantType\|class QScheme" $P/quark/torch/quantization/config/*.py | head
sed -n "$(grep -n 'class QTensorConfig' $P/quark/torch/quantization/config/config.py | cut -d: -f1),+40p" $P/quark/torch/quantization/config/config.py
