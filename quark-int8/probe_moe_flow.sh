V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== Quark int8 MoE apply ==="
sed -n '820,905p' $V/model_executor/layers/quantization/quark/quark_moe.py | grep -nE "def |topk|select_experts|self\.moe|kernel|moe_quant_config" | head -20
echo "=== RoutedExperts.forward / topk_ids ==="
grep -n "def forward\|def _forward\|topk_ids\|select_experts\|apply(" $V/model_executor/layers/fused_moe/routed_experts.py | head -30
