"""对拍：vLLM 在 gfx90a 上实际使用的 MoE 路由算子 vs 官方参考实现 Gate.forward 的语义。

参考（checkpoint 自带 inference/model.py:809-827）：
    scores = linear(x, W) / gate_temp(=1.0)
    scores = softplus(scores).sqrt()                  # score_func == sqrtsoftplus
    indices = (scores + correction_bias).topk(6)      # bias 只影响选择
    weights = scores.gather(indices)                  # 权重用**未加 bias**的值
    weights /= weights.sum(-1, keepdim=True) + 1e-20
    weights *= route_scale(=1.5)
"""
import os, sys, torch, torch.nn.functional as F

def reference(logits, W, bias, topk=6, route_scale=1.5):
    x = logits
    scores = F.linear(x.float(), W.float())            # gate_temp = 1.0
    scores = F.softplus(scores).sqrt()
    idx = (scores + bias.unsqueeze(0)).topk(topk, dim=-1)[1]
    w = scores.gather(1, idx)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return w * route_scale, idx

def main():
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias
    dev = "cuda:0"
    torch.manual_seed(0)

    # 用真实 checkpoint 的 layer-0 gate 权重（在 ignore 名单里，未量化）
    import json
    O = "/models"
    wm = json.load(open(f"{O}/model.safetensors.index.json"))["weight_map"]
    from safetensors import safe_open
    def get(name):
        with safe_open(f"{O}/{wm[name]}", framework="pt") as f:
            return f.get_tensor(name).float()
    W = get("layers.0.ffn.gate.weight").to(dev)
    bias = get("layers.0.ffn.gate.bias").to(dev).float()
    print(f"gate W{W.shape}  bias{bias.shape}  bias[min={bias.min():.4f} max={bias.max():.4f}]")

    n_experts = W.shape[0]
    fails = 0
    for M in (1, 19, 64, 256):
        # 现实量级：hidden 状态 absmax≈1.7（实测 h_absmax）
        h = (torch.randn(M, W.shape[1], device=dev) * 0.6)
        ref_w, ref_idx = reference(h, W, bias)

        # 走 vLLM 实际代码路径（它会调用 torch.ops._moe_C.topk_softplus_sqrt）
        got_w, got_idx = fused_topk_bias(
            hidden_states=h, gating_output=F.linear(h.float(), W.float()),
            scoring_func="sqrtsoftplus", e_score_correction_bias=bias,
            topk=6, renormalize=True, indices_type=torch.int32,
            routed_scaling_factor=1.5,
        )
        got_w = got_w.float()
        ids_same = bool(torch.equal(got_idx.long(), ref_idx.long()))
        # 顺序可能不同（参考是降序、算子未排序），按 id 对齐后比权重
        wref_sorted = torch.gather(ref_w, 1, torch.argsort(ref_idx, dim=1))
        wgot_sorted = torch.gather(got_w, 1, torch.argsort(got_idx.long(), dim=1))
        ids_sorted_same = bool(torch.equal(torch.sort(ref_idx, dim=1).values.long(),
                                          torch.sort(got_idx.long(), dim=1).values.long()))
        wdiff = float((wref_sorted - wgot_sorted).abs().max())
        rel = wdiff / float(ref_w.abs().max())
        ok = ids_sorted_same and rel < 1e-4
        print(f"  M={M:4d} 选中专家集合相同={ids_sorted_same} 同序={ids_same} "
              f"权重最大差={wdiff:.3e} (相对 {rel:.2e}) sum={got_w.sum(-1).mean():.4f} -> {'ok' if ok else 'BAD'}")
        if not ok:
            fails += 1
            print(f"      ref_idx[0]={ref_idx[0].tolist()}")
            print(f"      got_idx[0]={got_idx[0].tolist()}")
            print(f"      ref_w[0]={ref_w[0].tolist()}")
            print(f"      got_w[0]={got_w[0].tolist()}")
    print(f"\n结论：{'全部一致 ✅' if fails==0 else f'{fails} 例不一致 ❌'}")

main()
