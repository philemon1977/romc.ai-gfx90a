# -*- coding: utf-8 -*-
"""gfx90a indexer 内核 v2 验证：**用真实写入端**构造缓存，覆盖 next_n>1 与 SHUFFLE。

方法学修正（本次教训）：上一轮用自造的行主序缓存做等价性验证，掩盖了两个 bug
（q 的 next_n 寻址、SHUFFLE 布局）。本脚本一律用 indexer_k_quant_and_cache_triton
（生产写入端）生成缓存，并独立校验页内布局自洽性。
"""
import os, torch
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    rocm_fp8_paged_mqa_logits, fp8_paged_mqa_logits_torch,
    indexer_k_quant_and_cache_triton,
)
from vllm.v1.worker.workspace import init_workspace_manager
init_workspace_manager(torch.device("cuda"))
torch.manual_seed(0)

D, H, BS = 128, 32, 64
NB, NTOK, B = 8, 512, 2
dev = "cuda"
MML = 512

def build_cache(tokens):
    """用生产写入端把 tokens 写进 packed K 缓存（值区 + scale 区，block_size>1 ⇒ SHUFFLE）。"""
    kv = torch.zeros(NB, BS, D + 4, dtype=torch.uint8, device=dev)
    k = (torch.randn(tokens, D, device=dev) * 0.2).to(torch.float8_e4m3fn)
    slot = torch.arange(tokens, dtype=torch.int64, device=dev)   # 顺序落位
    indexer_k_quant_and_cache_triton(k, kv, slot, 128, None)
    return kv, k

def ref_shuffle(kv, q, w, ctx, bt, next_n, mml):
    """按写入端真实布局的 torch 参考（值区 SHUFFLE + 页内 scale 区）。"""
    kvf = kv.view(NB, -1)
    vals = kvf[:, : BS * D].view(torch.float8_e4m3fn).float()
    scales = kvf[:, BS * D :].contiguous().view(torch.float32)
    # 反 SHUFFLE：直接用写入端公式 gather（比 permute 推导可靠）
    tt = torch.arange(BS, device=kv.device)
    dd = torch.arange(D, device=kv.device)
    off = (
        (tt[:, None] // 16) * (16 * D)
        + (tt[:, None] % 16) * 16
        + (dd[None, :] // 16) * 256
        + (dd[None, :] % 16)
    )                                                        # [BS, D] 写入端偏移
    v = vals[:, off.reshape(-1)].view(NB, BS, D)             # [NB, BS, D] 行主序
    Bq, nn, Hh, _ = q.shape
    out = torch.full((Bq * nn, mml), float("-inf"), device=dev)
    qf = q.float()
    for b in range(Bq):
        c = int(ctx[b])
        npages = (c + BS - 1) // BS
        kk = torch.cat([v[bt[b, p]] for p in range(npages)])[:c]
        ss = torch.cat([scales[bt[b, p]] for p in range(npages)])[:c]
        for s in range(nn):
            qo = c - nn + s
            if qo < 0:
                continue
            sc_ = torch.relu(qf[b, s] @ kk.T) * w[b * nn + s][:, None]   # [H, c]
            lg = sc_.sum(0) * ss
            n = min(c, qo + 1)
            out[b * nn + s, :n] = lg[:n]
    return out

def cmp(a, b, tag):
    fin = torch.isfinite(a) & torch.isfinite(b)
    same = torch.equal(torch.isfinite(a), torch.isfinite(b))
    if fin.any():
        x, y = a[fin].float(), b[fin].float()
        cos = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
        print("  %-34s 掩码一致=%-5s 最大差=%.3e cos=%.6f" % (tag, same, (x - y).abs().max().item(), cos))
    else:
        print("  %-34s 无公共有限元" % tag)

kv, k = build_cache(NTOK)
# 布局自洽性：每位置 max|v| 应 ≈ 448 × scale（写入端 amax/448）
kvf = kv.view(NB, -1)
vals = kvf[:, : BS * D].view(torch.float8_e4m3fn).float()
scales = kvf[:, BS * D :].contiguous().view(torch.float32)
rowmajor = vals.view(NB, BS, D)
_tt = torch.arange(BS, device=dev); _dd = torch.arange(D, device=dev)
_off = ((_tt[:, None] // 16) * (16 * D) + (_tt[:, None] % 16) * 16 + (_dd[None, :] // 16) * 256 + (_dd[None, :] % 16)).reshape(-1)
unshuf = vals[:, _off].view(NB, BS, D)
for name, v in (("行主序", rowmajor), ("SHUFFLE(反解)", unshuf)):
    mx = v.abs().amax(dim=-1)                      # [NB, BS]
    pred = 448.0 * scales
    ratio = (mx / pred.clamp_min(1e-12)).median().item()
    print("  [布局自洽] %-14s 中位比 max|v|/(448*scale) = %.4f  （应≈1.0）" % (name, ratio))

q = (torch.randn(B, 1, H, D, device=dev) * 0.2).to(torch.float8_e4m3fn)
w = torch.rand(B, H, device=dev)
ctx = torch.tensor([256, 480], dtype=torch.int32, device=dev)
bt = torch.arange(NB, dtype=torch.int32, device=dev).view(1, -1).repeat(B, 1)
print("=== next_n = 1 ===")
ref1 = ref_shuffle(kv, q, w, ctx, bt, 1, MML)
ker1 = rocm_fp8_paged_mqa_logits(q, kv.view(NB, BS, 1, D + 4), w, ctx, bt, None, MML)
cmp(ker1, ref1, "kernel vs 真实布局参考")
cmp(fp8_paged_mqa_logits_torch(q, kv.view(NB, BS, 1, D + 4), w, ctx, bt, MML), ref1, "上游回退 vs 真实布局参考")
print("=== next_n = 3（Bug1 回归） ===")
q3 = (torch.randn(B, 3, H, D, device=dev) * 0.2).to(torch.float8_e4m3fn)
w3 = torch.rand(B * 3, H, device=dev)
ref3 = ref_shuffle(kv, q3, w3, ctx, bt, 3, MML)
ker3 = rocm_fp8_paged_mqa_logits(q3, kv.view(NB, BS, 1, D + 4), w3, ctx, bt, None, MML)
cmp(ker3, ref3, "kernel vs 真实布局参考")

print("=== 图捕获（含 next_n=1） ===")
try:
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            rocm_fp8_paged_mqa_logits(q, kv.view(NB, BS, 1, D + 4), w, ctx, bt, None, MML)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        cap = rocm_fp8_paged_mqa_logits(q, kv.view(NB, BS, 1, D + 4), w, ctx, bt, None, MML)
    g.replay(); torch.cuda.synchronize()
    print("  ✅ 捕获成功；replay 与即时执行一致:", bool(torch.equal(cap, ker1)))
except Exception as e:
    print("  ❌ 捕获失败:", type(e).__name__, str(e)[:180])
