V=/usr/local/lib/python3.12/dist-packages/vllm
python3 -c "import vllm; print('VLLM_VERSION:', vllm.__version__)"
python3 -c "import quark; print('QUARK_OK:', quark.__version__)" 2>&1 | tail -1
python3 -c "import aiter; print('AITER_OK:', getattr(aiter,'__version__','n/a'))" 2>&1 | tail -1
pip list 2>/dev/null | grep -iE "amd-quark|aiter|humming|onnxruntime" 
echo "=== int8 linear kernel selection ==="
ls $V/model_executor/kernels/linear/ 2>/dev/null
grep -rn "int8" $V/model_executor/kernels/linear/__init__.py 2>/dev/null | head -20
echo "=== AITER int8 kernel on rocm? ==="
grep -rln "AiterInt8\|w8a8_block_int8\|int8_paged\|gemm_a8w8\|w8a8." $V/model_executor/kernels/linear/ $V/_aiter_ops.py 2>/dev/null | head
grep -n "gfx9\|940\|906\|is_rocm\|_make_aiter" $V/_aiter_ops.py | head -30
echo "=== aiter .so for gfx90a int8 ==="
find /usr/local/lib/python3.12/dist-packages/aiter* -iname "*.so" 2>/dev/null | grep -i "gfx90a\|a8w8\|int8" | head -20
ls /usr/local/lib/python3.12/dist-packages/aiter/jit/ 2>/dev/null | grep -iE "gfx90a|int8|a8w8" | head -20
echo "=== quark scheme w8a8int8 kernel path in vllm ==="
sed -n '880,960p' $V/model_executor/layers/quantization/quark/quark.py
