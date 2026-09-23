# -*- coding: utf-8 -*-
"""决定性布局判据：用生产写入端写已知 k，再按两种寻址读回，谁恢复原值谁是对的。"""
import os, torch
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import indexer_k_quant_and_cache_triton
from vllm.v1.worker.workspace import init_workspace_manager
init_workspace_manager(torch.device("cuda"))
torch.manual_seed(0)
D, BS, NB = 128, 64, 8
NT = NB * BS
dev = "cuda"
kv = torch.zeros(NB, BS, D + 4, dtype=torch.uint8, device=dev)
k = (torch.randn(NT, D, device=dev) * 0.3).to(torch.float8_e4m3fn)
slot = torch.arange(NT, dtype=torch.int64, device=dev)
indexer_k_quant_and_cache_triton(k, kv, slot, 128, None)
kvf = kv.view(NB, -1)
vals = kvf[:, : BS * D].view(torch.float8_e4m3fn).float()          # [NB, BS*D]
scales = kvf[:, BS * D :].contiguous().view(torch.float32)          # [NB, BS]
orig = k.float().view(NB, BS, D)
tt = torch.arange(BS, device=dev); dd = torch.arange(D, device=dev)
shuf_off = ((tt[:, None] // 16) * (16 * D) + (tt[:, None] % 16) * 16
            + (dd[None, :] // 16) * 256 + (dd[None, :] % 16)).reshape(-1)
rowmajor = vals.view(NB, BS, D)
shuffled = vals[:, shuf_off].view(NB, BS, D)
def corr(a, b):
    x, y = a.reshape(-1), b.reshape(-1)
    return torch.corrcoef(torch.stack([x, y]))[0, 1].item()
print("=== 写入→读回 对比原值（0.99+ = 该寻址正确） ===")
print("  行主序        corr = %+.4f" % corr(rowmajor, orig))
print("  SHUFFLE 反解  corr = %+.4f" % corr(shuffled, orig))
print("=== scale 区自洽（每位置 scale ≈ amax/448） ===")
for nm, v in (("行主序", rowmajor), ("SHUFFLE", shuffled)):
    amax = v.abs().amax(dim=-1)
    ratio = (amax / (448.0 * scales)).median().item()
    print("  %-9s 中位比 amax/(448*scale) = %.4f" % (nm, ratio))
print("=== 参考：原始 k 的 amax/(448*scale) ===")
print("  原始         中位比 = %.4f" % (orig.abs().amax(dim=-1) / (448.0 * scales)).median().item())
