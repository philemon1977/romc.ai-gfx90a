# -*- coding: utf-8 -*-
"""隔离微基准：UVA(pinned 主机内存) 的 pin / 映射 / 拷贝 行为与吞吐。"""
import time, torch
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

def t(tag, fn):
    t0 = time.time(); r = fn(); dt = time.time() - t0
    print("  %-28s %7.3f s" % (tag, dt)); return r

GB = 1 << 30
print("=== 1) pin 1 GiB ===")
cpu = t("pin_memory 1 GiB", lambda: torch.empty(GB, dtype=torch.uint8).pin_memory())
print("  is_pinned =", cpu.is_pinned())
print("=== 2) UVA 映射 ===")
g = t("get_accelerator_view", lambda: get_accelerator_view_from_cpu_tensor(cpu))
print("  device =", g.device, " dtype =", g.dtype)
print("=== 3) 写 256 MiB 到 UVA 视图 ===")
src = torch.zeros(GB // 4, dtype=torch.uint8, device="cuda")
t("copy_ 256MiB (GPU->UVA)", lambda: (g[: src.numel()].copy_(src), torch.cuda.synchronize()))
print("=== 4) 从 UVA 视图读回 256 MiB ===")
t("to(cuda) 256MiB (UVA->GPU)", lambda: (g[: src.numel()].to("cuda"), torch.cuda.synchronize()))
print("=== 5) 小规模索引读（模拟 engram lookup 形态） ===")
idx = torch.randint(0, GB, (8 * 512,), dtype=torch.long, device="cuda")
t("index_select 4096 行 (UVA)", lambda: (g[idx], torch.cuda.synchronize()))
print("  有效带宽 = %.2f GB/s" % ((8 * 512) / (time.time() - t0) / 1e9))
print("=== 6) 图捕获中读 UVA ===")
try:
    gr = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        out = g[idx].to("cuda")
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    with torch.cuda.graph(gr):
        out2 = g[idx].to("cuda")
    gr.replay(); torch.cuda.synchronize()
    print("  ✅ 捕获成功，replay 与即时一致:", bool(torch.equal(out, out2)))
except Exception as e:
    print("  ❌ 捕获失败:", type(e).__name__, str(e)[:180])
