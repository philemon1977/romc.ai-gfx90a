import torch
print("torch", torch.__version__, "| hip/rocm:", getattr(torch.version, "hip", None) or getattr(torch.version, "hsa", None))
assert torch.cuda.is_available(), "no GPU visible to torch!"
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} = {p.name} gfx{p.gcnArchName.split(':')[0].replace('gfx','')} {p.total_memory//(1<<30)}GiB")
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
c = (a @ a).float().sum().item()
print("matmul 4096^3 fp16 ok, checksum:", round(c, 1))
print("PYTORCH: PASS")
