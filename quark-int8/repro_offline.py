#!/usr/bin/env python3
"""Offline repro of the real decode-MoE call, from the captured tensors.

Question it answers definitively: on the REAL call, does the upstream kernel or my GEMV kernel
match an independent reference (dequantize the int4 weights, then per-pair GEMV using
token = pair // top_k and expert = topk_ids[pair])?

Whichever side disagrees tells us where the semantic mismatch is, and we can iterate here in
seconds instead of restarting the server every 7 minutes.
"""
import torch

CAP = "/tmp/mi250_moe_real.pt"
GROUP = 128


def main():
    d = torch.load(CAP, map_location="cuda")
    A, B, S = d["A"].cuda(), d["B"].cuda(), d["B_scale"].cuda()
    C_mine, C_up = d["C_mine"].cuda(), d["C_up"].cuda()
    ids, tw = d["topk_ids"], d["topk_weights"]
    ids = ids.cuda() if ids is not None else None
    ids_flat = ids.reshape(-1) if ids is not None else None
    tw = tw.cuda() if tw is not None else None
    M, N, K, top_k = d["M"], d["N"], d["K"], d["top_k"]
    print(f"captured: A{A.shape} B{tuple(B.shape)} S{tuple(S.shape)} C{C_mine.shape} "
          f"M={M} N={N} K={K} top_k={top_k} bm={d['block_m']} num_valid={d['num_valid']}")
    print(f"topk_ids{tuple(ids.shape) if ids is not None else None} "
          f"topk_weights{tuple(tw.shape) if tw is not None else None} sorted[0:8]={d['sorted'][:8].tolist()} "
          f"eids[0:8]={d['expert_ids'][:8].tolist()}")

    # 参考：只对 A 里真实的行（捕获了前 Ntok 行）算，pair p -> token p//top_k
    Ntok = A.shape[0]
    pairs = Ntok * top_k
    used = torch.unique(ids_flat[:pairs]).tolist()
    print(f"A 捕获 {Ntok} 行 => 可校验 {pairs} 对；涉及 {len(used)} 个不同专家")

    # 反量化（offset-binary: value = nibble - 8），只做用到的专家
    wb = B[used].to(torch.int64)                      # [E_u, N, K//2]
    nib = torch.empty(len(used), N, K, dtype=torch.float32, device="cuda")
    for j in range(2):
        nib[:, :, j::2] = ((wb >> (4 * j)) & 0xF).to(torch.float32) - 8.0
    sc = S[used].to(torch.float32).repeat_interleave(GROUP, dim=2)
    wdq = nib * sc                                    # [E_u, N, K]
    del nib

    x = A.to(torch.float32)                           # [Ntok, K]
    ref = torch.empty(pairs, N, dtype=torch.float32, device="cuda")
    remap = {e: i for i, e in enumerate(used)}
    for p in range(pairs):
        ref[p] = wdq[remap[int(ids_flat[p])]] @ x[p // top_k]
    if tw is not None and tw.numel() >= pairs:
        ref = ref * tw.reshape(-1)[:pairs, None].float()

    me = C_mine.reshape(-1, N)[:pairs].float()
    up = C_up.reshape(-1, N)[:pairs].float()
    den = ref.abs().mean().item() + 1e-6

    def err(a):
        return (a - ref).abs().mean().item() / den * 100

    print(f"\n参考 |均值|={den:.4f}")
    print(f"  我的 GEMV  vs 参考: {err(me):8.2f}%   |均值|={me.abs().mean().item():.4f}")
    print(f"  上游内核   vs 参考: {err(up):8.2f}%   |均值|={up.abs().mean().item():.4f}")
    print(f"  我 vs 上游        : {(me-up).abs().mean().item()/den*100:8.2f}%")

    # 排查方向：若上游对而我不对 -> 我的 (token,expert) 映射错；反之亦然
    if err(up) < 5 and err(me) > 5:
        print("\n⇒ 上游符合参考、我不符合 ⇒ **我的映射错**，接着验证是否为 'sorted 布局' 语义：")
        sid, eid, bm = d["sorted"].cuda(), d["expert_ids"].cuda(), d["block_m"]
        # 上游口径: token = sorted_slot_value // top_k, expert = eids[slot // bm]
        pairs_up = torch.full((pairs,), -1, dtype=torch.int64, device="cuda")
        n_slots = min(sid.numel(), eid.numel() * bm)
        s = sid[:n_slots].to(torch.int64)
        e = eid.repeat_interleave(bm)[:n_slots].to(torch.int64)
        ok = (s >= 0) & (s < pairs)
        pairs_up[s[ok]] = e[ok]
        same = int((pairs_up[:pairs] == ids_flat[:pairs].to(torch.int64)).sum())
        print(f"   sorted 反推出的专家 与 topk_ids 一致的对数: {same}/{pairs}")
        t_up = (torch.arange(pairs, device='cuda') // top_k)
        print(f"   （上游 token 映射 = pair//top_k 一致？是，两边都用同一公式）")
        mism = (pairs_up[:pairs] != ids_flat[:pairs].to(torch.int64)).nonzero().flatten()
        if mism.numel():
            k = mism[0].item()
            print(f"   首个不一致 pair={k}: sorted推得={int(pairs_up[k])} topk_ids={int(ids_flat[k])}")
            print(f"   注意：sorted 数组可能只覆盖部分 slot（numel={sid.numel()}，eids={eid.numel()}×bm={bm}）")
    elif err(me) < 5 and err(up) > 5:
        print("\n⇒ 我符合参考、上游不符合 ⇒ 上游用的是另一套映射（我的集成需要改成那一套）")
    else:
        print("\n⇒ 两边都不符合参考 ⇒ 参考或捕获口径有误（例如 topk_ids 与本次调用不同步）")


if __name__ == "__main__":
    main()
