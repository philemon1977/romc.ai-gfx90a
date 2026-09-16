V=/usr/local/lib/python3.12/dist-packages/vllm
echo "VLLM_VERSION: $(python3 -c 'import vllm;print(vllm.__version__)')"
python3 -c "import quark;print('QUARK_OK',quark.__version__)" 2>&1|tail -1
python3 -c "import aiter;print('AITER_OK')" 2>&1|tail -1
echo "=== scaled_mm int8 kernels ==="
ls $V/model_executor/kernels/linear/scaled_mm/ 2>/dev/null
grep -rn "Int8" $V/model_executor/kernels/linear/scaled_mm/*.py 2>/dev/null | grep -iE "rocm|aiter|class" | head -20
echo "=== int8 oracle ==="
ls $V/model_executor/kernels/linear/ ; find $V/model_executor/kernels -name "*.py" | xargs grep -ln "select.*int8\|Int8*Kernel" 2>/dev/null | head
echo "=== aiter int8 gemm live test on gfx90a ==="
python3 - <<'PY'
import torch
torch.cuda.init()
print("device:", torch.cuda.get_device_name(0))
try:
    import aiter
    M,K,N=256,512,1024
    x=torch.randint(-127,128,(M,K),device='cuda',dtype=torch.int8)
    w=torch.randint(-127,128,(N,K),device='cuda',dtype=torch.int8)
    sx=torch.rand(M,1,device='cuda')/100
    sw=torch.rand(1,N,device='cuda')/100
    out=torch.empty(M,N,device='cuda',dtype=torch.float32)
    r=aiter.gemm_a8w8(x,w,sx,sw,out)
    torch.cuda.synchronize()
    ref=(x.float()@w.float().t())*(sx@sw)
    print("aiter.gemm_a8w8 OK, max rel err:", ((out-ref).abs()/(ref.abs()+1e-3)).max().item())
except Exception as e:
    import traceback; traceback.print_exc()
    print("AITER_INT8_FAIL:", type(e).__name__, str(e)[:300])
print("=== torch._int_mm fallback test ===")
try:
    x=torch.randint(-127,128,(256,512),device='cuda',dtype=torch.int8)
    w=torch.randint(-127,128,(512,1024),device='cuda',dtype=torch.int8)
    y=torch._int_mm(x,w); torch.cuda.synchronize()
    print("torch._int_mm OK", y.dtype, y.shape)
except Exception as e:
    print("INT_MM_FAIL:", str(e)[:200])
print("=== triton availability ===")
import triton; print("triton", triton.__version__)
PY
echo "=== humming availability ==="
python3 -c "import humming; print('humming OK')" 2>&1 | tail -1
