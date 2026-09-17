V=/usr/local/lib/python3.12/dist-packages/vllm
echo "############ RoutedExperts.forward / forward_modular ############"
sed -n '1160,1300p' $V/model_executor/layers/fused_moe/routed_experts.py
