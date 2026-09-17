P=/usr/local/lib/python3.12/dist-packages
V=$P/vllm
echo "===== quark-cli help (stderr stripped) ====="
quark-cli --help 2>/dev/null | head -25
echo "===== quark-cli entry ====="
grep -rn "quark-cli\|console_scripts" $P/quark-*/entry_points.txt /usr/local/bin/quark-cli 2>/dev/null | head -5
echo "===== file2file public API ====="
grep -n "^def \|^class " $P/quark/torch/quantization/file2file_quantization.py
echo "===== qwen3_5.py structure ====="
grep -n "class \|FusedMoE(\|MergedColumnParallelLinear\|RowParallelLinear\|ColumnParallelLinear\|shared_expert\|in_proj\|out_proj\|quant_config\|exclude\|re:compile" $V/model_executor/models/qwen3_5.py | head -70
echo "===== quark.py exclude mechanism ====="
grep -n "exclude" $V/model_executor/layers/quantization/quark/quark.py | head -20
echo "===== quark int8 linear scale suffix ====="
grep -n "weight_scale\|_scale\b" $V/model_executor/layers/quantization/quark/quark.py | sed -n '1,25p'
