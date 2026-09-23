#!/usr/bin/env python3
"""离线分析 [MOECHK] 捕获的真实 MoE 张量（/tmp/moe_real_capture.pt）。

回答：
  1. B 的专家维度 = 本 rank 专家数还是全局专家数？expert_map 是否生效？
     （决定 topk_ids 能不能直接索引 B —— 也决定我们那个 GEMV 内核用全局号索引是否合法）
  2. 上游融合 Triton WNA16 核在 gfx90a 上算出来的 C，与"显式反量化 + matmul"的
     参考差多少？（参考已在 CPU 上对暴力实现验证为 0 误差）
  3. SwiGLU clamp 假设的量化：真实 MoE 输入 A 经过真权重 w13 后的 pre-activation，
     有多大比例超出 ±10（checkpoint 的训练设定是夹到 ±10；上游没夹）。
"""
import sys
import torch

p = sys.argv[1] if len(sys.argv) > 1 else "/tmp/moe_real_capture.pt"
d = torch.load(p, map_location="cpu", weights_only=False)
print("=== 捕获内容 ===")
for k, v in d.items():
    if hasattr(v, "shape"):
        print(f"  {k:18s} {str(tuple(v.shape)):22s} {v.dtype}")
    else:
        print(f"  {k:18s} {v}")

B, Bs, A = d["B"], d["B_scale"], d["A"]
g = int(d["group"]); N = int(d["N"]); K = int(d["K"]); top_k = int(d["top_k"])
ri = d["real_idx"].to(torch.int64)

print(f"\n=== 1. 专家分片事实 ===")
print(f"  B.shape[0] = {B.shape[0]}  (本 rank 专家数 or 全局专家数？)")
print(f"  N = {N} (每 rank 的中间维×2?)， K = {K}， group = {g}")
print(f"  expert_ids 覆盖: min={int(d['expert_ids'].min())} max={int(d['expert_ids'].max())}")
if "topk_weights" in d and d["topk_weights"] is not None:
    w = d["topk_weights"].float()
    print(f"  topk_weights sum: min={w.sum(-1).min():.4f} max={w.sum(-1).max():.4f} (期望 ≈ route_scale 1.5)")

print(f"\n=== 2. 上游 MoE 内核 vs 显式反量化参考（真实对 {ri.numel()} 个）===")
ref = d["ref"]; C = d["C"].view(-1, N).float()[: ref.shape[0]]
a, b = C[ri], ref[ri]
rel = (a - b).abs().mean().item() / (b.abs().mean().item() + 1e-9) * 100
cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
print(f"  相对误差 = {rel:.3f}%   cos = {cos:.6f}   maxabs = {(a-b).abs().max():.4e}")
print(f"  |上游| 均值 = {a.abs().mean():.4f}  |参考| 均值 = {b.abs().mean():.4f}")
print(f"  判据：<1% ⇒ 上游 int4 反量化在 gfx90a 上是对的；>5% ⇒ 这就是根因")
if "C_mine" in d:
    pass

if d.get("mul_routed_weight") is False:
    print(f"\n=== 3. SwiGLU clamp 假设（仅 gemm1 的激活性）===")
    half = N // 2
    for e in torch.unique(d["expert_ids"].to(torch.int64))[:4].tolist():
        be = B[e]
        lo = (be & 0x0F).float(); hi = ((be >> 4) & 0x0F).float()
        codes = torch.empty(N, K); codes[:, 0::2] = lo - 8; codes[:, 1::2] = hi - 8
        W = codes * Bs[e].float().repeat_interleave(g, dim=1)
        pre = A.float() @ W.t()          # [T, N]
        gate, up = pre[:, :half], pre[:, half:]
        print(f"  expert {e}: gate |x|>10 比例={float((gate.abs()>10).float().mean())*100:.2f}% "
              f"max|gate|={gate.abs().max():.2f} | up |x|>10 比例={float((up.abs()>10).float().mean())*100:.2f}% "
              f"max|up|={up.abs().max():.2f}  (gate/up 的 std={gate.std():.2f}/{up.std():.2f})")
    print("  若比例接近 0 ⇒ clamp 缺失无关紧要；若可观（>1%）⇒ 确实换了激活函数")
