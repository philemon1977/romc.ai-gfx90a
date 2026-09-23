#!/bin/bash
P=/usr/local/lib/python3.12/dist-packages/vllm
echo "== MoE WNA16 内核支持的量化类型 =="
for f in $(grep -rl "invoke_fused_moe_wna16" $P/model_executor/layers/fused_moe/ 2>/dev/null | head -3); do
  echo "--- $(basename $f)"
  grep -n "uint4b8\|uint8b128\|SUPPORTED\|num_bits\|def invoke_fused_moe_wna16" "$f" | head -12
done
echo
echo "== moe_wna16 量化方法支持的位宽 =="
grep -rn "uint4b8\|uint8b128" $P/model_executor/layers/quantization/moe_wna16.py 2>/dev/null | head -6
echo
echo "== experts_int8 走什么内核（gfx90a 上是否可用）=="
grep -rn "class .*ExpertsInt8\|aiter\|triton\|cutlass" $P/model_executor/layers/quantization/experts_int8.py 2>/dev/null | head -10
