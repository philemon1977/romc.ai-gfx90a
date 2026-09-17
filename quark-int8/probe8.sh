P=/usr/local/lib/python3.12/dist-packages
echo "===== quantize_model_per_safetensor signature ====="
sed -n '1611,1700p' $P/quark/torch/quantization/file2file_quantization.py
echo "===== single stage quantize (per-channel on 3D?) ====="
sed -n '1133,1230p' $P/quark/torch/quantization/file2file_quantization.py
