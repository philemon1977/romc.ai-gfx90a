#!/usr/bin/env python3
"""探针 A：实测单 GCD 可达 HBM 带宽。

目的：验证「27.9 GB / 1.6 TB/s ≈ 17 ms/步」这个 roofline 分母在本机是否成立。
模型无关，但本仓库只用它解释 dense INT8 模型的 conc-64 缺口。

三个负载，覆盖不同访问模式：
  1. sum   —— 纯读（最接近 decode 读权重的模式）
  2. copy  —— 读+写（分母 = 2 × 字节数）
  3. gemv  —— A @ x，读权重为主、写极小（decode 的形状）

用法（容器内）：ROCR_VISIBLE_DEVICES=0 python3 probe_hbm_bw.py [GiB]
"""

import sys
import time

import torch

GiB = 1024**3
size_gib = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
nbytes = int(size_gib * GiB)

print(f"torch {torch.__version__}  device={torch.cuda.get_device_name(0)}")
print(f"tensor size: {size_gib:.1f} GiB (fp16)")

nel = nbytes // 2
A = torch.empty(nel, dtype=torch.float16, device="cuda").normal_()
B = torch.empty_like(A)

# --- 1. 纯读 ---
for _ in range(2):
    A.sum(dtype=torch.float32)
torch.cuda.synchronize()
t = time.time()
A.sum(dtype=torch.float32)
torch.cuda.synchronize()
dt = time.time() - t
print(f"sum   : {dt*1000:8.1f} ms  -> {nbytes/dt/1e9:7.1f} GB/s (read)")

# --- 2. 读+写 ---
for _ in range(2):
    B.copy_(A)
torch.cuda.synchronize()
t = time.time()
B.copy_(A)
torch.cuda.synchronize()
dt = time.time() - t
print(f"copy  : {dt*1000:8.1f} ms  -> {2*nbytes/dt/1e9:7.1f} GB/s (rd+wr)")

# --- 3. GEMV（decode 形状：读权重为主）---
K = 8192
N = nel // K
W = A[: N * K].view(N, K)
x = torch.randn(K, dtype=torch.float16, device="cuda")
for _ in range(2):
    W @ x
torch.cuda.synchronize()
t = time.time()
W @ x
torch.cuda.synchronize()
dt = time.time() - t
print(f"gemv  : {dt*1000:8.1f} ms  -> {W.numel()*2/dt/1e9:7.1f} GB/s (weight read)  shape={tuple(W.shape)}")

del A, B, W, x
torch.cuda.empty_cache()
