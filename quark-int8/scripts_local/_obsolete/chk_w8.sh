#!/bin/bash
P=/usr/local/lib/python3.12/dist-packages/vllm
echo "== mixed_precision 内核清单 =="
ls $P/model_executor/kernels/linear/mixed_precision/
echo
echo "== 各内核支持的 quant types =="
for f in $P/model_executor/kernels/linear/mixed_precision/*.py; do
  t=$(grep -m1 -A6 "SUPPORTED_QUANT" "$f" 2>/dev/null | tr -d ' \n' | cut -c1-140)
  [ -n "$t" ] && echo "  $(basename $f): $t"
done
echo
echo "== CT WNA16 支持的 num_bits 映射 =="
grep -rn "WNA16_SUPPORTED_TYPES_MAP\s*=" -A 8 $P/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py | head -14
echo
echo "== ROCm 上 MPLinearKernel 的选择顺序 =="
grep -n "class .*LinearKernel\|def get_kernel\|is_rocm\|priority" $P/model_executor/kernels/linear/MPLinearKernel.py | head -20
