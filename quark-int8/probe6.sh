P=/usr/local/lib/python3.12/dist-packages
V=$P/vllm
echo "===== quark-cli --help ====="
quark-cli --help 2>&1 | head -30
echo "===== file2file_quantization.py head ====="
sed -n '1,60p' $P/quark/torch/quantization/file2file_quantization.py
echo "===== tensor_quantize.py symbols ====="
grep -n "^def \|^class " $P/quark/torch/quantization/tensor_quantize.py | head
echo "===== quark_moe int8 create_weights (cont) ====="
sed -n '760,850p' $V/model_executor/layers/quantization/quark/quark_moe.py
echo "===== qwen3_5.py structure ====="
grep -n "class \|FusedMoE\|MergedColumnParallelLinear\|RowParallelLinear\|ColumnParallelLinear\|shared_expert\|in_proj\|out_proj\|quant_config\|weight_scale" $V/model_executor/models/qwen3_5.py | head -60
echo "===== quark.py exclude mechanism ====="
grep -n "exclude" $V/model_executor/layers/quantization/quark/quark.py | head -20
echo "===== weight_scale suffix names quark expects ====="
grep -rn "weight_scale\|input_scale" $V/model_executor/layers/quantization/quark/quark.py | grep -n "suffix\|name" | head -12
