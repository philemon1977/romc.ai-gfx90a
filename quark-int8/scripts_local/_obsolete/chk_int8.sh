#!/bin/bash
P=/usr/local/lib/python3.12/dist-packages/vllm
echo "== 1) CT 是否支持 4/8bit 混合（config_groups 多组）=="
grep -n "config_groups" $P/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py | head -6
echo
echo "== 2) CT WNA16 支持的 num_bits / group_size =="
grep -rn "num_bits\|TRITON_W4A16_SUPPORTED\|SUPPORTED_GROUP" $P/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py 2>/dev/null | head -12
echo
echo "== 3) 8bit 权重走哪个 kernel（是否存在 W8A16/TritonWNA16-8bit）=="
ls $P/model_executor/kernels/linear/ 2>/dev/null | head; ls $P/model_executor/kernels/linear/mixed_precision/ 2>/dev/null
grep -rn "uint8b128\|uint4b8\|int8" $P/model_executor/kernels/linear/mixed_precision/*.py 2>/dev/null | head -8
echo
echo "== 4) MoE WNA16 路径支持的位宽（专家 down 走这里）=="
grep -rn "uint4b8\|uint8b128\|num_bits" $P/model_executor/layers/fused_moe/experts/triton_moe.py 2>/dev/null | head -8
