python3 - <<'PY'
import torch, time
import aiter
M,K,N=256,512,1024
x=torch.randint(-127,128,(M,K),device='cuda',dtype=torch.int8)
w=torch.randint(-127,128,(N,K),device='cuda',dtype=torch.int8)
sx=torch.rand(M,1,device='cuda')/100
sw=torch.rand(1,N,device='cuda')/100
out=torch.empty(M,N,device='cuda',dtype=torch.float32)
t0=time.time()
aiter.gemm_a8w8(x,w,sx,sw,out)
torch.cuda.synchronize()
print("gemm_a8w8 gfx90a build+run OK in %.1fs"%(time.time()-t0))
ref=(x.float()@w.float().t())*(sx@sw)
print("max rel err:", ((out-ref).abs()/(ref.abs()+1e-3)).max().item())
PY
