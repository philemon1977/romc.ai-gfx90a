#!/bin/bash
P=/usr/local/lib/python3.12/dist-packages/vllm
echo "== CT 是否支持 mxfp4 格式（含 MoE）=="
grep -rn "mxfp4\|float_quant\|scale_dtype" $P/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py | head -10
echo
echo "== CT 的 MoE 方案里有 mxfp4 吗 =="
ls $P/model_executor/layers/quantization/compressed_tensors/schemes/ | head -20
echo
echo "== vLLM 原生 mxfp4 走的 MoE 内核（今晚健康臂用的就是它）=="
grep -rn "class .*Mxfp4MoE\|TRITON_UNFUSED\|def get_mxfp4_backend" $P/model_executor/layers/quantization/mxfp4.py | head -8
