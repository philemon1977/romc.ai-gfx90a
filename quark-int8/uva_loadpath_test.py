# -*- coding: utf-8 -*-
"""隔离验证：CPU 切片 -> UVA 视图切片 的 copy_（engram 装载路径的确切形态）。"""
import time, torch
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

def timed(tag, fn, limit=60):
    t0 = time.time(); r = fn(); dt = time.time() - t0
    print("  %-42s %8.3f s" % (tag, dt)); return r

GB = 1 << 30
N = 2 * GB                       # 2 GiB 表（1 层的一半，缩小版）
print("=== 准备 ===")
cpu = timed("pin %d MiB" % (N // 2**20), lambda: torch.empty(N, dtype=torch.uint8).pin_memory())
view = timed("UVA 视图", lambda: get_accelerator_view_from_cpu_tensor(cpu))
print("  view.device=%s  is_pinned(src)=%s" % (view.device, cpu.is_pinned()))

print("=== A) 普通 CPU 张量 -> UVA 视图整体 copy_ ===")
src = torch.zeros(N, dtype=torch.uint8)
timed("copy_ 2 GiB (CPU -> UVA)", lambda: (view.copy_(src), torch.cuda.synchronize()), 120)

print("=== B) CPU 切片 -> UVA 视图切片（装载路径的确切形态） ===")
seg = src[: GB // 2]
timed("view[0:n].copy_(seg 512 MiB)",
      lambda: (view[0 : seg.numel()].copy_(seg), torch.cuda.synchronize()), 120)

print("=== C) 非零偏移的切片（第二部分） ===")
timed("view[off:off+n].copy_(seg)",
      lambda: (view[GB : GB + seg.numel()].copy_(seg), torch.cuda.synchronize()), 120)

print("=== D) 读回校验 ===")
chk = torch.zeros(16, dtype=torch.uint8, device="cuda")
chk.copy_(view[:16]); torch.cuda.synchronize()
print("  ✅ 读回成功:", bool((chk.cpu() == 0).all()))
