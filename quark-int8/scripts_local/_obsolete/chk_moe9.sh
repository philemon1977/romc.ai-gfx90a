#!/bin/bash
P=/usr/local/lib/python3.12/dist-packages/vllm
echo "== triton_moe.py SUPPORTED_W 区块 =="
sed -n '590,612p' $P/model_executor/layers/fused_moe/experts/triton_moe.py
echo
echo "== experts_int8.py 关键实现（前 60 行）=="
sed -n '1,60p' $P/model_executor/layers/quantization/experts_int8.py
