# SPDX-License-Identifier: Apache-2.0
"""MI250X GEMV-style decode MoE for the W4A16 (uint8-packed int4) path.

Why: measured at M=6 tokens/step (MTP(5)) on this model, vLLM's padded Triton WNA16 MoE
costs 246.0 us/layer => 14.8 ms/step, i.e. ~40% of a 37 ms step, while the HBM bandwidth
floor is ~2.0 ms/step. The padding waste is structural: BM=16 with ~1 real token per active
expert => ~16x wasted M, and the grid collapses to ~40 blocks on 104 CUs.

This kernel removes the M padding entirely: one program per (token, expert) pair x N-tile,
a true GEMV (no tl.dot), fp32 accumulation. Measured vs the incumbent at this model's
shapes with realistic routing (59/57 distinct experts):
    gemm1 (gate+up): 185.1 -> 114.4 us (+62%)   numerics 2.58% (fp32 acc vs bf16 partials)
    gemm2 (down)   :  60.9 ->  37.7 us (+62%)   numerics 0.14%
=> ~93.9 us/layer saved => ~5.6 ms/step => ~+18% single-stream TPS (projected).

Runtime layout (pinned from vllm .../fused_moe/oracle/int_wna16.py:1630, compressed-tensors):
    B     : uint8 [E, N, K//2]   each byte = 2 int4, LOW nibble = even k, value = nibble - 8
    B_scale: [E, N, K//group]    group = 128 for this checkpoint
    A     : bf16 [M, K] (gemm1) / [M*topk, K] (gemm2)
Gate: MI250_MOE_GEMV=1 (default 0 -> upstream kernel untouched).

★★ 2026-09-20 v3（scale 提出循环）—— 4 倍量级的修正，取代 v1/v2 的写法 ★★
微基准（quark-int8/moe_gemv_diag.py，单 GCD，真实 checkpoint 布局 [E,N,K//2]）：
    全量 v1/v2          1674.5 us   60.1 GB/s
    去掉 scale 乘法       378.2 us  266.2 GB/s   <- 4.4x，说明瓶颈是 scale 的取用
    去掉 nibble 提取      1663.9 us   60.5 GB/s   <- nibble 解码几乎免费
    纯 load（同访存模式）  112-162 us 620-898 GB/s <- 访存模式本身能到 900 GB/s
被证伪的两个猜测（留痕，别再走一遍）：
  ① "tl.sum(axis=1) 每个 k 步都跨 lane 归约是主因" —— 二维累加器版（v2）只快 1.01x；
  ② "纯 ALU 受限" —— 实测 0.9 Tops/s，仅 fp32 峰值的 4%。
真因：v1 的 scale 下标是逐元素的 kk//GROUP（kk = k0 + 2c + j），Triton 无法向量化，
每个 k 步退化成 [BLOCK_N, BLOCK_K//2] 次 2 字节 gather（BLOCK_K=64 时 2048 次取指，
其中只有 2 个不同地址），把访存指令数放大约 16 倍。
v3：BLOCK_K = G_PER_STEP*GROUP，三维累加器 [BLOCK_N, G_PER_STEP, GROUP//2]，循环内不碰
scale；每步收尾归约一次第三维、只乘 [BLOCK_N, G_PER_STEP] 的 scale 切片。
实测（同一次 sweep）：gemm1 1694.5 -> 412.3 us（4.1x），gemm2 891.1 -> 208.5 us（4.3x），
对拍 fp32 参考 0.14%（与 v1/v2 同级，即精度不变）。
开关：MI250_MOE_GEMV_KERNEL=v3（默认）/ v1。
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

GROUP = 128  # default (Ornith / CT gs=128); the real value is read from
             # block_shape[1] at call time so gs=32 checkpoints work too
_ENABLE = os.environ.get("MI250_MOE_GEMV", "0").strip() not in ("0", "", "false", "False")
_DEBUG = os.environ.get("MI250_MOE_GEMV_DEBUG", "0").strip() not in ("0", "", "false", "False")
# v3 = scale 提出 k 循环（gemm1 4.1x / gemm2 4.3x）；设 MI250_MOE_GEMV_KERNEL=v1 可退回旧内核
_V3 = os.environ.get("MI250_MOE_GEMV_KERNEL", "v3").strip().lower() not in ("v1", "1", "old")
_seen = {"takeover": 0, "skip": 0}
# apply() 里有现成的 topk_ids；暂存下来即可，完全不需要从 sorted 布局反推（那一层是上一个失败的原因）
_current = {"ids": None}


def set_current_topk(ids):
    _current["ids"] = ids


def clear_current_topk():
    _current["ids"] = None
_orig_call = None


@triton.jit
def _gemv_moe_k(X, W, S, O, IDS, WTS, M, K, N, TOPK,
                APPLY_W: tl.constexpr, GROUP: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """One program per (token, expert) PAIR x N-tile: no M padding, true GEMV, fp32 acc."""
    pid_p = tl.program_id(0)
    pid_n = tl.program_id(1)
    t = pid_p // TOPK
    e = tl.maximum(tl.load(IDS + pid_p), 0)   # padding 行的 topk_ids 可能是 -1：钳制以免负偏移
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k2 = tl.arange(0, BLOCK_K // 2)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        wb = tl.load(W + e * N * (K // 2) + offs_n[:, None] * (K // 2) + (k0 // 2 + offs_k2)[None, :])
        wb = wb.to(tl.int32)
        part = tl.zeros((BLOCK_N,), dtype=tl.float32)
        # ★ gfx90a 修正（gs=32 泛化）：scale 必须**逐 k 元素**取 `k // GROUP`，
        #   不能每个 BLOCK_K 只取一个 `k0 // GROUP`。原写法只在 BLOCK_K == GROUP
        #   （Ornith gs=128 配 BLOCK_K=128）时才对；本模型 gs=32 时一个 BLOCK_K=128
        #   跨 4 个 group，另外 3/4 的 k 会乘错 scale ⇒ MoE 输出整体失真 ⇒ 生成退化
        #   （实测：17*19 只吐 "	 **	 **..."）。上游参考内核也是按元素取
        #   `offs_k // group_size`，这里对齐。
        for j in tl.static_range(2):
            kk = k0 + offs_k2 * 2 + j
            sc = tl.load(
                S + e * N * (K // GROUP) + offs_n[:, None] * (K // GROUP)
                + (kk // GROUP)[None, :]
            )
            nib = ((wb >> (4 * j)) & 0xF) - 8
            xj = tl.load(X + t * K + kk)
            part += tl.sum(
                nib.to(tl.float32) * xj[None, :].to(tl.float32) * sc.to(tl.float32),
                axis=1,
            )
        acc += part
    if APPLY_W:
        acc = acc * tl.load(WTS + pid_p)
    tl.store(O + pid_p * N + offs_n, acc.to(O.dtype.element_ty))


@triton.jit
def _gemv_moe_v3(X, W, S, O, IDS, WTS, M, K, N, TOPK,
                 APPLY_W: tl.constexpr, GROUP: tl.constexpr, G_PER_STEP: tl.constexpr,
                 BLOCK_N: tl.constexpr):
    """v3：BLOCK_K = G_PER_STEP * GROUP；循环内完全不碰 scale。

    三维累加器 [BLOCK_N, G_PER_STEP, GROUP//2]，每个 k 步收尾把第三维归约掉、只乘一次
    [BLOCK_N, G_PER_STEP] 的 scale 切片（同一行内这 G 个值在内存里连续，可向量化）。
    见文件头 ★★ 段：这一改动实测 gemm1 4.1x / gemm2 4.3x。
    """
    pid_p = tl.program_id(0)
    pid_n = tl.program_id(1)
    t = pid_p // TOPK
    e = tl.maximum(tl.load(IDS + pid_p), 0)   # padding 行可能是 -1：钳制以免负偏移
    HALF: tl.constexpr = GROUP // 2           # 每 group 的字节数（gs=32 -> 16）
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_g = tl.arange(0, G_PER_STEP)
    offs_h = tl.arange(0, HALF)
    row_w = offs_n[:, None, None] * (K // 2)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, G_PER_STEP * GROUP):
        acc3 = tl.zeros((BLOCK_N, G_PER_STEP, HALF), dtype=tl.float32)
        for j in tl.static_range(2):
            wb = tl.load(W + e * N * (K // 2) + row_w
                         + (k0 // 2 + offs_g[None, :, None] * HALF + offs_h[None, None, :]))
            xj = tl.load(X + t * K + (k0 + offs_g[:, None] * GROUP + offs_h[None, :] * 2 + j))
            acc3 += (((wb.to(tl.int32) >> (4 * j)) & 0xF) - 8).to(tl.float32) \
                * xj[None, :, :].to(tl.float32)
        sg = tl.load(S + e * N * (K // GROUP) + offs_n[:, None] * (K // GROUP)
                     + (k0 // GROUP + offs_g)[None, :])
        acc += tl.sum(tl.sum(acc3, axis=2) * sg.to(tl.float32), axis=1)
    if APPLY_W:
        acc = acc * tl.load(WTS + pid_p)
    tl.store(O + pid_p * N + offs_n, acc.to(O.dtype.element_ty))


def _v3_cfg(K, N, group, is_gemm2, pairs=8):
    """按形状选 (BLOCK_N, G_PER_STEP, num_warps)；(0,0,0) 表示 v3 不可用。

    两套实测（moe_gemv_v3_bench.py 全 N；moe_gemv_prod_bench.py 生产分片形状）：
      · 全 N（N=4096，网格 512 program）：BN=64 G=8 w=4 最优，275 GB/s；
      · 生产分片（N_local=512/768 ⇒ 网格只剩 128/192 个 program，严重欠占用）：
        **BN=16 G=8 w=1** 最优 —— gemm1 45.2 us（v1 是 391.4 us，8.7x）、
        gemm2 26.5 us（v1 是 132.9 us，5.0x）。小 BLOCK_N 换来更多 program，
        比"每 program 干得多"重要得多。
      · pairs>128（M>16）时 BN=64 G=4 w=1 反超 3%（pairs=256: 499.7 vs 515.1 us）。
    """
    gps = 8
    while gps > 1 and ((gps * group > K) or (K % (gps * group))):
        gps //= 2
    if K % (gps * group):
        return 0, 0, 0
    if N > 1024:                      # 未分片（每 rank 持全部 N）：按全 N 那份调优
        for bn in (64, 128, 32):
            if bn <= N and N % bn == 0:
                return bn, gps, 4
        return 0, 0, 0
    if pairs > 128:
        gps = min(gps, 4)
        pref = (64, 32, 16)
    else:
        pref = (16, 32, 64)
    for bn in pref:
        if bn <= N and N % bn == 0:
            return bn, gps, 1
    return 0, 0, 0


@triton.jit
def _invert_sorted_k(SORTED, EIDS, OUT, S, NUM_VALID,
                     BLOCK_M: tl.constexpr, BLOCK: tl.constexpr):
    """out[sorted_token_ids[i]] = expert_ids[i // BLOCK_M]  — one launch, not five torch ops."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < S
    s = tl.load(SORTED + offs, mask=m, other=NUM_VALID)
    e = tl.load(EIDS + offs // BLOCK_M, mask=m, other=0)
    ok = m & (s >= 0) & (s < NUM_VALID)
    tl.store(OUT + s, e, mask=ok)


def _num_valid(sorted_token_ids, pairs):
    """vLLM pads the sorted array with the constant ``num_valid_tokens`` (real pairs), which is
    < pairs when A is padded to a cudagraph batch size. Padding entries sit at the END of each
    expert's block, and valid entries are strictly smaller, so max() recovers it in one
    reduction (torch.unique cost ~170 us and ate the whole win)."""
    try:
        pad = int(sorted_token_ids.max().item())
    except Exception:
        return pairs
    return pad if 0 <= pad <= pairs else pairs


def _expert_of_pair(sorted_token_ids, expert_ids, pairs, block_m, num_valid):
    """Invert vLLM's sorted layout into a per-pair expert id (one tiny Triton launch).

    Only slots with ``sorted < num_valid`` are real pairs; the rest are padding and must NOT
    be written, otherwise a padding block's expert lands on a real pair index.
    """
    out = torch.zeros(pairs, dtype=torch.int32, device=sorted_token_ids.device)
    S = int(sorted_token_ids.numel())
    BLOCK = 1024
    _invert_sorted_k[(triton.cdiv(S, BLOCK),)](
        sorted_token_ids, expert_ids, out, S, num_valid, BLOCK_M=block_m, BLOCK=BLOCK)
    return out


def invoke_gemv_wna16(
    A, B, C, B_scale, B_zp, topk_weights, sorted_token_ids, expert_ids,
    num_tokens_post_padded, mul_routed_weight, top_k, config, compute_type,
    use_int8_w8a16, use_int4_w4a16, block_shape,
):
    """Drop-in replacement for vLLM's invoke_fused_moe_wna16_triton_kernel (int4 path only).

    Returns True when it took over, False to let the upstream kernel run.
    """
    if not (_ENABLE and use_int4_w4a16 and not use_int8_w8a16 and B_zp is None
            and B.dtype == torch.uint8 and A.dtype == torch.bfloat16
            # group size comes from the checkpoint's block_shape; must divide K and
            # be a power of two so `k0 // group` is exact for every BLOCK_K step
            and block_shape is not None and block_shape[1] > 0
            and block_shape[1] & (block_shape[1] - 1) == 0
            and A.shape[1] % block_shape[1] == 0
            and B.shape[2] * 2 == A.shape[1]):
        _seen["skip"] += 1
        return False

    M, K = A.shape
    N = B.shape[1]
    pairs = M * top_k
    block_m = config["BLOCK_SIZE_M"]
    if pairs == 0 or N % 128 != 0:
        return False
    # ★ 只在 decode 小 M 区接管：prefill（如 M=2048、pairs=20480）必须留给上游的 tl.dot 分块，
    #   否则① 输出会被算坏（NLL 爆炸）② 那条调用慢得多，把 decode 的收益全抵消。
    if M > 64 or pairs > 256:
        _seen["skip"] += 1
        return False

    # gemm2 的内核太小（~38 us），逆置换的固定开销会吃掉收益 => 默认只接管 gemm1
    if mul_routed_weight and os.environ.get("MI250_MOE_GEMV_BOTH", "0") not in ("1", "true", "yes"):
        _seen["skip"] += 1
        return False
    if _DEBUG and _seen["takeover"] == 0:
        print(f"[MI250_MOE_GEMV-DUMP] A{tuple(A.shape)} stride={A.stride()} contig={A.is_contiguous()} "
              f"C{tuple(C.shape)} stride={C.stride()} contig={C.is_contiguous()} "
              f"B{tuple(B.shape)} Bs{tuple(B_scale.shape)} sorted[0:12]={sorted_token_ids[:12].tolist()} "
              f"eids[0:6]={expert_ids[:6].tolist()} nblocks={expert_ids.numel()} "
              f"numel_sorted={sorted_token_ids.numel()} pairs={pairs} top_k={top_k} bm={block_m}", flush=True)
    cur = _current.get("ids")
    zero_tail = False
    if cur is not None and cur.numel() > 0:
        # ★ 直接吃 apply() 里的 topk_ids：一次 view，零成本，且不受 padding 路由语义影响
        ids = cur.reshape(-1)[:pairs].contiguous()
        num_valid = ids.numel()
        zero_tail = num_valid < pairs
    else:
        num_valid = _num_valid(sorted_token_ids, pairs)
        ids = _expert_of_pair(sorted_token_ids, expert_ids, pairs, block_m, num_valid)
    if _DEBUG and _seen["takeover"] == 0:
        real = (ids[:num_valid] >= 0)
        n_real = int(real.sum())
        print(f"[MI250_MOE_GEMV-GATE] M={M} pairs={pairs} num_valid={num_valid} "
              f"real_pairs={n_real} distinct_all={int(torch.unique(ids[:num_valid]).numel())} "
              f"distinct_real={int(torch.unique(ids[:num_valid][real]).numel()) if n_real else 0}", flush=True)
        # 自检：同一次真实调用上跑上游内核并与我的结果对拍（找集成层差异的最快手段）
        try:
            c_up = torch.zeros_like(C)
            _orig_call(A, B, c_up, B_scale, B_zp, topk_weights, sorted_token_ids, expert_ids,
                       num_tokens_post_padded, mul_routed_weight, top_k, config, compute_type,
                       use_int8_w8a16, use_int4_w4a16, block_shape)
            mine = C.view(-1, N)[:num_valid].float()
            up = c_up.view(-1, N)[:num_valid].float()

            def _err(a, b):
                return (a - b).abs().mean().item() / (b.abs().mean().item() + 1e-6) * 100

            m = (ids[:num_valid] >= 0)
            n_real = int(m.sum())
            msg = f"[MI250_MOE_GEMV-SELFCHECK] 全体 {num_valid} 对: {_err(mine, up):.2f}%"
            if n_real:
                msg += f" | ★真实对 {n_real} 对: {_err(mine[m], up[m]):.2f}% | 上游真实对|均值|={up[m].abs().mean().item():.3f}"
            else:
                msg += " | 无 ids>=0 的标记（padding 可能不是 -1）"
            print(msg, flush=True)
            # ★ 把真实张量存盘：之后在离线环境复现，避免再花 7 分钟/轮的服务器循环
            try:
                torch.save({"A": A[:min(M, 8)].clone(), "B": B.clone(), "B_scale": B_scale.clone(),
                            "C_mine": C[:min(M, 8)].clone(), "C_up": c_up[:min(M, 8)].clone(),
                            "topk_ids": cur.clone() if cur is not None else None,
                            "topk_weights": (topk_weights.clone() if topk_weights is not None else None),
                            "sorted": sorted_token_ids[:2048].clone(),
                            "expert_ids": expert_ids[:64].clone(),
                            "config": dict(config), "top_k": top_k, "M": M, "N": N, "K": K,
                            "block_m": block_m, "num_valid": num_valid},
                           "/tmp/mi250_moe_real.pt")
                print("[MI250_MOE_GEMV-CAPTURE] saved /tmp/mi250_moe_real.pt", flush=True)
            except Exception as exc:
                print(f"[MI250_MOE_GEMV-CAPTURE] failed: {exc!r}", flush=True)
        except Exception as exc:
            print(f"[MI250_MOE_GEMV-SELFCHECK] 失败: {exc!r}", flush=True)
    wts = topk_weights.reshape(-1) if (mul_routed_weight and topk_weights is not None) else None
    if mul_routed_weight and wts is None:
        return False
    if wts is not None and wts.numel() != pairs:
        return False

    group = int(block_shape[1])
    out = C.view(-1, N)[:pairs]
    if zero_tail:
        out[num_valid:].zero_()          # padding 行：上游写 0，这里对齐
    dummy = ids if wts is None else wts

    # ★ v3（scale 提出 k 循环）优先；形状不满足时退回 v1
    if _V3:
        bn, gps, nw = _v3_cfg(K, N, group, bool(mul_routed_weight), pairs)
        if bn:
            grid = (num_valid, N // bn)
            _gemv_moe_v3[grid](
                A, B, B_scale, out, ids, dummy, M, K, N, top_k,
                APPLY_W=(wts is not None), GROUP=group, G_PER_STEP=gps,
                BLOCK_N=bn, num_warps=nw,
            )
            _seen["takeover"] += 1
            if _DEBUG and _seen["takeover"] % 200 == 1:
                print(f"[MI250_MOE_GEMV] v3 takeover={_seen['takeover']} skip={_seen['skip']} "
                      f"M={M} K={K} N={N} pairs={pairs} gs={group} BN={bn} G={gps} w={nw} "
                      f"apply_w={wts is not None}", flush=True)
            return True

    BLOCK_N = 128
    # BLOCK_K 必须是 2 的幂；scale 已按逐元素 `k // group` 取，所以 BLOCK_K 与
    # group 是否整除**不再**是正确性前提（但保持 128 以沿用同一份调优）。
    BLOCK_K = max(group, 128)
    if BLOCK_K % group != 0 or BLOCK_K & (BLOCK_K - 1) != 0:
        return False
    grid = (num_valid, N // 128)
    _gemv_moe_k[grid](
        A, B, B_scale, out, ids, dummy, M, K, N, top_k,
        APPLY_W=(wts is not None), GROUP=group, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    _seen["takeover"] += 1
    if _DEBUG and _seen["takeover"] % 200 == 1:
        print(f"[MI250_MOE_GEMV] v1 takeover={_seen['takeover']} skip={_seen['skip']} "
              f"M={M} K={K} N={N} pairs={pairs} gs={group} apply_w={wts is not None}", flush=True)
    return True
