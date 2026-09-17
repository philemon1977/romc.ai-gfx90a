P=/usr/local/lib/python3.12/dist-packages
echo "===== f2f_weight_converters consumers ====="
grep -rn "f2f_weight_converters" $P/quark --include=*.py | grep -v __pycache__ | grep -v template.py
echo "===== LLMTemplate.get / to QConfig ====="
grep -n "def get\b\|def get(\|exclude_layers_name\|def get_qconfig\|def to_qconfig\|def quant_config" $P/quark/torch/quantization/config/template.py | head -20
echo "===== quark-cli commands ====="
quark-cli --help 2>&1 | grep -vE "QUARK-WARNING|^\s*$|onnx" | head -30
echo "===== Int8 per-channel spec ====="
grep -n "class Int8PerTensorSpec\|class Int8PerChannel\|Int8PerChannelSpec" $P/quark/torch/quantization/config/*.py $P/quark/torch/quantization/config/**/*.py 2>/dev/null | head -5
echo "===== vllm maybe_fuse_shared_experts ====="
grep -rn "def maybe_fuse_shared_experts\|def is_model_fused_shared_expert_compatible" $P/vllm --include=*.py | grep -v __pycache__
F=$(grep -rln "def maybe_fuse_shared_experts" $P/vllm --include=*.py | grep -v __pycache__ | head -1)
grep -n "def maybe_fuse_shared_experts" $F | cut -d: -f1 | xargs -I{} sed -n '{},+60p' $F
