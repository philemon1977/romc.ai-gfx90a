P=/usr/local/lib/python3.12/dist-packages
F=$P/quark/torch/quantization/file2file_quantization.py
echo "===== fused MoE mentions in file2file ====="
grep -n "experts\|gate_up\|fuse\|3D\|ndim\|dim() == 3\|preshard" $F | head -40
echo "===== _resolve_presharded_chunk_rows ====="
sed -n '805,826p' $F
echo "===== quantize_and_save shard: 3D handling ====="
sed -n '1420,1520p' $F
echo "===== weight_convert.py classes ====="
grep -n "^class \|^def " $P/quark/torch/quantization/weight_convert.py | head
echo "===== examples/tests calling quantize_model_per_safetensor ====="
grep -rln "quantize_model_per_safetensor" $P --include=*.py | head
echo "===== QConfig.to_dict + QLayerConfig signature ====="
sed -n "$(grep -n 'class QLayerConfig' $P/quark/torch/quantization/config/config.py | cut -d: -f1),+30p" $P/quark/torch/quantization/config/config.py
