P=/usr/local/lib/python3.12/dist-packages
V=$P/vllm
echo "===== aiter int8 scaled_mm kernel impl ====="
sed -n '25,160p' $V/model_executor/kernels/linear/scaled_mm/aiter.py
echo "===== ROCm selection priority in scaled_mm/__init__ ====="
grep -n "aiter\|Aiter\|ROCM\|rocm\|TRITON\|int8" $V/model_executor/kernels/linear/scaled_mm/__init__.py | head -40
