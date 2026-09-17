#!/usr/bin/env python3
"""我的 GEMV vs vLLM 现役 gemm1 内核，同输入直接对拍（去掉我自己的参考实现这一变量）。

统一到 vLLM 的真实布局：B 以 uint8 字节视图 [E, N, K//2] 传入（每字节 2 个 int4，低 nibble=偶 k），
真值 = nibble - 8。M=1/TOPK=1 使 MoE 退化为单次 GEMV，便于逐元素比对。
"""
import torch, triton.language as tl
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_wna16_triton_kernel
from gemv_moe import run_ours

E, N, K, GROUP, PACK = 8, 64, 256, 128, 8
HIDDEN = K
CFG = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1,
       "SPLIT_K": 1, "num_warps": 4, "num_stages": 2}
dev = "cuda"

torch.manual_seed(0)
x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
S = (torch.rand(E, N, K // GROUP, dtype=torch.float32, device=dev) + 0.5).to(torch.bfloat16)
w_log = torch.randint(-8, 8, (E, N, K), dtype=torch.int64, device=dev)

# 按 vLLM 的真实约定打包成 uint8 字节视图: byte m 的低 nibble = k=2m, 高 nibble = k=2m+1
wbytes = torch.zeros(E, N, K // 2, dtype=torch.uint8, device=dev)
lo = ((w_log[:, :, 0::2] + 8) & 0xF).to(torch.uint8)
hi = ((w_log[:, :, 1::2] + 8) & 0xF).to(torch.uint8)
wbytes = (lo | (hi << 4)).contiguous()
# 内核接口取 int32 视图（与 vLLM 线上一致：int32 [E,N,K//8] 的底层字节就是上面的布局）
w_i32 = wbytes.view(torch.int32)

# --- 现役内核 ---
c = torch.zeros(1, 1, N, dtype=torch.bfloat16, device=dev)
sid = torch.zeros(CFG["BLOCK_SIZE_M"], dtype=torch.int32, device=dev)
eid = torch.zeros(1, dtype=torch.int32, device=dev)
npp = torch.tensor([CFG["BLOCK_SIZE_M"]], dtype=torch.int32, device=dev)
invoke_fused_moe_wna16_triton_kernel(x, w_i32, c, S, None, torch.ones(1, 1, device=dev),
                                     sid, eid, npp, False, 1, CFG, tl.bfloat16, False, True, [0, GROUP])
inc = c[0, 0].float()

# --- 我的内核（int32 视图入口，8 nibble/word，k=8i+j）---
ids = torch.zeros(1, dtype=torch.int32, device=dev)
wts = torch.ones(1, dtype=torch.float32, device=dev)
mine = run_ours(x, w_i32, S, ids, wts, N, 1, BLOCK_N=64, BLOCK_K=128).float()[0]

den = inc.abs().mean().item() + 1e-6
print(f"现役   |out|均值={inc.abs().mean().item():8.4f}")
print(f"我的   |out|均值={mine.abs().mean().item():8.4f}")
print(f"两者相对误差 = {(mine-inc).abs().mean().item()/den*100:.2f}%   "
      f"{'✅ 一致 ⇒ 我的内核正确、是我的参考实现写错了' if (mine-inc).abs().mean().item()/den < 0.05 else '❌ 不一致 ⇒ 我的内核有 bug'}")
