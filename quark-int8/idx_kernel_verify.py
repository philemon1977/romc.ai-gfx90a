# -*- coding: utf-8 -*-
"""自研 indexer 内核的组合验证（落库版）——覆盖此前未复跑的部分：
  1) next_n ∈ {1, 3}：与 SHUFFLE 真值参考逐元素对比（Bug1 回归）
  2) 图捕获：replay 与即时执行逐位一致
  3) 分派：DSV41_IDX_AITER_KERNEL=1 时确实走自研内核（用"内核输出 == SHUFFLE 参考"间接判定）
方法学：一律用**生产写入端**构造缓存（indexer_k_quant_and_cache_triton），不用自造布局。
"""
import os, sys, torch
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    rocm_fp8_paged_mqa_logits, indexer_k_quant_and_cache_triton,
)
from vllm.v1.worker.workspace import init_workspace_manager
init_workspace_manager(torch.device("cuda"))
torch.manual_seed(0)

D, BS, NB, H, B, MML = 128, 64, 8, 32, 2, 512
NT, dev = NB * BS, "cuda"
kv = torch.zeros(NB, BS, D + 4, dtype=torch.uint8, device=dev)
k = (torch.randn(NT, D, device=dev) * 0.3).to(torch.float8_e4m3fn)
indexer_k_quant_and_cache_triton(k, kv, torch.arange(NT, dtype=torch.int64, device=dev), 128, None)
kv4 = kv.view(NB, BS, 1, D + 4)
kvf = kv.view(NB, -1)
vals = kvf[:, : BS * D].view(torch.float8_e4m3fn).float()
scales = kvf[:, BS * D :].contiguous().view(torch.float32)
tt = torch.arange(BS, device=dev); dd = torch.arange(D, device=dev)
shuf = ((tt[:, None] // 16) * (16 * D) + (tt[:, None] % 16) * 16
        + (dd[None, :] // 16) * 256 + (dd[None, :] % 16)).reshape(-1)
kd = vals[:, shuf].view(NB, BS, D)          # 反解出的 k（SHUFFLE）

def ref(q, w, ctx, bt, next_n):
    out = torch.full((B * next_n, MML), float("-inf"), device=dev)
    for b in range(B):
        c = int(ctx[b]); npx = (c + BS - 1) // BS
        kcat = torch.cat([kd[bt[b, p]] for p in range(npx)])[:c]
        scat = torch.cat([scales[bt[b, p]] for p in range(npx)])[:c]
        for s in range(next_n):
            qo = c - next_n + s
            if qo < 0:
                continue
            lg = (torch.relu(q.float()[b, s] @ kcat.T) * w[b * next_n + s][:, None]).sum(0) * scat
            n = min(c, qo + 1)
            out[b * next_n + s, :n] = lg[:n]
    return out

def cmp(a, b, tag):
    fin = torch.isfinite(a) & torch.isfinite(b)
    same = torch.equal(torch.isfinite(a), torch.isfinite(b))
    x, y = a[fin].float(), b[fin].float()
    cos = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
    print("  %-34s 掩码一致=%-5s 最大差=%.2e cos=%.6f %s" % (tag, same, (x - y).abs().max().item(), cos, "OK" if cos > 0.99999 and same else "❌"))
    return cos > 0.99999 and same

ctx = torch.tensor([256, 480], dtype=torch.int32, device=dev)
bt = torch.arange(NB, dtype=torch.int32, device=dev).view(1, -1).repeat(B, 1)
ok = True
for nn_ in (1, 3):
    q = (torch.randn(B, nn_, H, D, device=dev) * 0.3).to(torch.float8_e4m3fn)
    w = torch.rand(B * nn_, H, device=dev)
    ker = rocm_fp8_paged_mqa_logits(q, kv4, w, ctx, bt, None, MML)
    ok &= cmp(ker, ref(q, w, ctx, bt, nn_), "next_n=%d 内核 vs SHUFFLE 参考" % nn_)
print("=== 图捕获（next_n=1） ===")
q = (torch.randn(B, 1, H, D, device=dev) * 0.3).to(torch.float8_e4m3fn)
w = torch.rand(B, H, device=dev)
try:
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            base = rocm_fp8_paged_mqa_logits(q, kv4, w, ctx, bt, None, MML)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        cap = rocm_fp8_paged_mqa_logits(q, kv4, w, ctx, bt, None, MML)
    g.replay(); torch.cuda.synchronize()
    same = torch.equal(cap, base)
    print("  ✅ 捕获成功；replay 与即时执行逐位一致:", same)
    ok &= bool(same)
except Exception as e:
    print("  ❌ 捕获失败:", type(e).__name__, str(e)[:180]); ok = False
print("=== 结论：%s ===" % ("全部通过" if ok else "有失败"))
sys.exit(0 if ok else 1)
