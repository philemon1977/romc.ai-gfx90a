"""gfx90a (CDNA2 / MI250X) flash-decoding paged attention for decode-only steps.

WHY THIS EXISTS
---------------
Measured on this box (Ornith-1.5-397B-FP8, TP8, official InferenceX client):

    single-stream decode TPOT = 45.4 ms @1k  -> 171.1 ms @32k -> 553.9 ms @128k
                                                -> 967.9 ms @240k
    excess is STRICTLY linear in context:  TPOT ~= 45.4 + 3.97 ms x (ctx/1024)

At 128k the per-token KV read is ~2.0 GiB/die and it takes 0.509 s, i.e. **4 GiB/s
= 0.30% of the ~1.3 TB/s HBM peak of one MI250X GCD**. The cause is structural: a
decode step has ONE query token per sequence, this model has ``num_key_value_heads
= 2`` (so with TP8 each rank ends up with one KV head), and only 15 of the 60
layers are ``full_attention``. One (sequence, KV head) pair = one workgroup =
**1 of 104 CUs on the GCD doing anything** while it walks 248 KV blocks serially.
That is exactly what flash-decoding removes: split the KV range over many
workgroups, each returning partial ``(acc, running_max, running_sum)``, then merge
with a log-sum-exp reduction.

DESIGN NOTES (the parts that are easy to get wrong)
---------------------------------------------------
* GQA-aware tiling. ``n_q_per_kv = num_q_heads // num_kv_heads`` rows share one KV
  stream, so the M dimension of the dot is that group, padded to ``GROUP_PAD``
  (16) because MFMA on CDNA wants >=16x16x16 tiles. We are bandwidth- not
  compute-bound, so the padding waste is free.
* Layout-agnostic by construction: every tensor is addressed through explicit
  strides, so it binds to whichever paged layout vLLM's ``kv_cache_config`` ends
  up using (NHD ``[blocks, bs, kv_heads, dim]``, HND ``[blocks, kv_heads, bs,
  dim]``, or the CK ``[blocks, kv_heads, dim//x, bs, x]`` form after a view).
* Arbitrary block size supported. This model runs with the *hybrid* attention
  block size of 528 tokens (vLLM raises it so the attention page matches the GDN
  page), which no power-of-two ``BLOCK_N`` divides. So block lookup is done
  per-token with integer div/mod instead of assuming ``BLOCK_N | block_size``.
* Softcap / logits-soft-capping is NOT applied (Qwen3.5 attention does not use
  it); q/k RMSNorm happens upstream, this kernel consumes post-norm tensors.
* Numerics: partials and the reduction are fp32; ``exp`` on the shifted scores.
  Tolerance for the correctness harness is set from a bf16 reference, not guessed.

Two entry points: ``paged_flash_decode`` (split + merge, what you want) and
``paged_decode_no_split`` (the deliberate 1-workgroup baseline that reproduces the
pathology, kept so the gain is measured against the same code path, not against a
different algorithm).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
# split 1: partial attention over one KV range of one (seq, kv head)
# --------------------------------------------------------------------------- #


@triton.jit
def _decode_split_kernel(
    Q,  # [b, qh, d]
    K,  # [num_blocks, bs, kvh, d] addressed by strides
    V,
    OP,  # partial out  [b, qh, ns, d] fp32
    MP,  # partial max  [b, qh, ns]     fp32
    LP,  # partial sum  [b, qh, ns]     fp32
    BT,  # block_table  [b, max_blocks] int32/int64
    CTX,  # ctx_lens     [b]            int32/int64
    stride_qb,
    stride_qh,
    stride_kblk,
    stride_kb,
    stride_kh,
    stride_vblk,
    stride_vb,
    stride_vh,
    stride_opb,
    stride_oph,
    stride_ops,
    stride_mpb,
    stride_mph,
    stride_btb,
    sm_scale,
    n_q_per_kv: tl.constexpr,
    NUM_KV: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    split = pid % NUM_SPLITS
    rest = pid // NUM_SPLITS
    kv_head = rest % NUM_KV
    b = rest // NUM_KV

    ctx = tl.load(CTX + b).to(tl.int64)
    # half-open token range this program owns
    span = tl.cdiv(ctx, NUM_SPLITS)
    tok_start = split * span
    tok_end = tl.minimum(tok_start + span, ctx)

    rows = tl.arange(0, GROUP_PAD)  # q rows of this KV head's GQA group
    row_ok = rows < n_q_per_kv
    dd = tl.arange(0, HEAD_DIM)

    # q tile: [GROUP_PAD, HEAD_DIM]; kv head index == kv_head
    qh_index = kv_head * n_q_per_kv + rows
    q_ptrs = Q + b * stride_qb + qh_index[:, None] * stride_qh + dd[None, :]
    q = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0)

    m_i = tl.full([GROUP_PAD], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([GROUP_PAD], dtype=tl.float32)
    acc = tl.zeros([GROUP_PAD, HEAD_DIM], dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_N)
    for start in range(tok_start, tok_end, BLOCK_N):
        n = start + offs_n
        n_ok = n < tok_end
        # per-token block id -> supports block sizes that are not a multiple of BLOCK_N
        blk = tl.load(BT + b * stride_btb + (n // BLOCK_SIZE), mask=n_ok, other=0).to(tl.int64)
        in_blk = n % BLOCK_SIZE

        k_ptrs = K + blk[:, None] * stride_kblk + in_blk[:, None] * stride_kb + \
            kv_head * stride_kh + dd[None, :]
        k = tl.load(k_ptrs, mask=n_ok[:, None], other=0.0)  # [BLOCK_N, HEAD_DIM]

        qk = tl.dot(q, tl.trans(k)) * sm_scale            # [GROUP_PAD, BLOCK_N]
        qk = tl.where(n_ok[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V + blk[:, None] * stride_vblk + in_blk[:, None] * stride_vb + \
            kv_head * stride_vh + dd[None, :]
        v = tl.load(v_ptrs, mask=n_ok[:, None], other=0.0)  # [BLOCK_N, HEAD_DIM]
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    # store unnormalized partial; the merge does the softmax bookkeeping
    op_ptrs = OP + b * stride_opb + (kv_head * n_q_per_kv + rows)[:, None] * \
        stride_oph + split * stride_ops + dd[None, :]
    tl.store(op_ptrs, acc, mask=row_ok[:, None])
    tl.store(MP + b * stride_mpb + (kv_head * n_q_per_kv + rows) * stride_mph + split,
             m_i, mask=row_ok)
    tl.store(LP + b * stride_mpb + (kv_head * n_q_per_kv + rows) * stride_mph + split,
             l_i, mask=row_ok)


# --------------------------------------------------------------------------- #
# split 2: merge partials across splits (log-sum-exp)
# --------------------------------------------------------------------------- #


@triton.jit
def _merge_kernel(
    OP, MP, LP, OUT,
    stride_opb, stride_oph, stride_ops,
    stride_mpb, stride_mph,
    stride_ob, stride_oh,
    n_q_heads: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SPLIT_POW2: tl.constexpr,
    QH_POW2: tl.constexpr,
):
    b = tl.program_id(0)

    rows = tl.arange(0, QH_POW2)          # every local q head in one program
    row_ok = rows < n_q_heads
    dd = tl.arange(0, HEAD_DIM)
    s = tl.arange(0, SPLIT_POW2)
    s_ok = s < NUM_SPLITS

    qh_index = rows
    base_m = MP + b * stride_mpb + qh_index[:, None] * stride_mph + s[None, :]
    m = tl.load(base_m, mask=row_ok[:, None] & s_ok[None, :], other=float("-inf"))
    l = tl.load(LP + b * stride_mpb + qh_index[:, None] * stride_mph + s[None, :],
                mask=row_ok[:, None] & s_ok[None, :], other=0.0)

    m_max = tl.max(m, axis=1)
    w = tl.exp(m - m_max[:, None])          # [GROUP_PAD, SPLIT_POW2]
    denom = tl.sum(w * tl.where(s_ok[None, :], l, 0.0), axis=1)

    op_ptrs = OP + b * stride_opb + qh_index[:, None, None] * stride_oph + \
        s[None, :, None] * stride_ops + dd[None, None, :]
    acc = tl.load(op_ptrs, mask=row_ok[:, None, None] & s_ok[None, :, None], other=0.0)
    out = tl.sum(acc * w[:, :, None], axis=1) / tl.maximum(denom[:, None], 1e-30)

    tl.store(OUT + b * stride_ob + qh_index[:, None] * stride_oh + dd[None, :],
             out.to(OUT.dtype.element_ty), mask=row_ok[:, None])


# --------------------------------------------------------------------------- #
# baseline without splitting: reproduces the 1-workgroup pathology
# --------------------------------------------------------------------------- #


@triton.jit
def _decode_nosplit_kernel(
    Q, K, V, OUT, BT, CTX,
    stride_qb, stride_qh, stride_kblk, stride_kb, stride_kh,
    stride_vblk, stride_vb, stride_vh, stride_ob, stride_oh, stride_btb,
    sm_scale,
    n_q_per_kv: tl.constexpr, GROUP_PAD: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    kv_head = tl.program_id(1)
    ctx = tl.load(CTX + b).to(tl.int64)
    rows = tl.arange(0, GROUP_PAD)
    row_ok = rows < n_q_per_kv
    dd = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    qh_index = kv_head * n_q_per_kv + rows
    q = tl.load(Q + b * stride_qb + qh_index[:, None] * stride_qh + dd[None, :],
                mask=row_ok[:, None], other=0.0)

    m_i = tl.full([GROUP_PAD], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([GROUP_PAD], dtype=tl.float32)
    acc = tl.zeros([GROUP_PAD, HEAD_DIM], dtype=tl.float32)

    for start in range(0, ctx, BLOCK_N):
        n = start + offs_n
        n_ok = n < ctx
        blk = tl.load(BT + b * stride_btb + (n // BLOCK_SIZE), mask=n_ok, other=0).to(tl.int64)
        in_blk = n % BLOCK_SIZE
        k = tl.load(K + blk[:, None] * stride_kblk + in_blk[:, None] * stride_kb +
                    kv_head * stride_kh + dd[None, :], mask=n_ok[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(n_ok[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(V + blk[:, None] * stride_vblk + in_blk[:, None] * stride_vb +
                    kv_head * stride_vh + dd[None, :], mask=n_ok[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    out = acc / tl.maximum(l_i, 1e-30)[:, None]
    tl.store(OUT + b * stride_ob + qh_index[:, None] * stride_oh + dd[None, :],
             out.to(OUT.dtype.element_ty), mask=row_ok[:, None])


# --------------------------------------------------------------------------- #
# python API
# --------------------------------------------------------------------------- #


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1)).bit_length()


def num_splits_for(ctx: int, *, target_programs: int = 104, min_tokens: int = 512,
                   max_splits: int = 128) -> int:
    """How many KV splits to use for a batch at ``ctx`` tokens.

    One GCD of MI250X has 104 CUs; with 1 sequence x 1 KV head per rank the split
    count IS the parallelism, so aim at filling the device but never give a split
    fewer than ``min_tokens`` (merge traffic and per-program fixed cost dominate
    below that).
    """
    by_ctx = max(1, ctx // min_tokens)
    return max(1, min(target_programs, by_ctx, max_splits))


def paged_flash_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    ctx_lens: torch.Tensor,
    *,
    num_kv_heads: int,
    num_splits: int | None = None,
    block_size: int | None = None,
    sm_scale: float | None = None,
    block_n: int = 64,
    group_pad: int = 16,
    num_warps: int = 4,
    num_stages: int = 2,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode attention with split-KV (flash-decoding) over a paged KV cache.

    Args:
        q: ``[batch, num_q_heads, head_dim]`` (bf16/fp16), post q-norm.
        k_cache, v_cache: paged cache, any layout expressible with the strides
            ``(block, token_in_block, kv_head, dim)`` -- i.e. ``k_cache`` viewed as
            ``[num_blocks, block_size, num_kv_heads, head_dim]`` or the HND form.
        block_table: ``[batch, max_blocks]`` physical block ids.
        ctx_lens: ``[batch]`` sequence length including the current token.
        num_kv_heads: KV heads present *in this tensor* (after TP sharding).
        num_splits: override; default from :func:`num_splits_for`.

    Returns:
        ``[batch, num_q_heads, head_dim]`` in q's dtype.
    """
    assert q.dim() == 3, f"expected [b, qh, d], got {tuple(q.shape)}"
    # require the last dim contiguous so dd arange indexing is valid
    for t in (q, k_cache, v_cache):
        assert t.stride(-1) == 1, "last dim must be contiguous"
    bsz, n_q_heads, head_dim = q.shape
    n_kv = int(num_kv_heads)
    assert n_q_heads % n_kv == 0, "GQA grouping must divide evenly"
    n_per_kv = n_q_heads // n_kv
    blk = int(block_size) if block_size else int(k_cache.shape[1])
    ctx = int(ctx_lens.max().item())
    ns = int(num_splits) if num_splits else num_splits_for(ctx, target_programs=104)
    scale = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(head_dim)
    assert head_dim & (head_dim - 1) == 0, "HEAD_DIM must be a Triton constexpr pow2"
    assert n_per_kv <= group_pad, f"group {n_per_kv} > GROUP_PAD {group_pad}"
    assert _next_pow2(n_q_heads) <= 128, "q-head tile too wide for one program"

    out = torch.empty_like(q) if out is None else out
    dev = q.device
    op = torch.empty((bsz, n_q_heads, ns, head_dim), device=dev, dtype=torch.float32)
    mp = torch.empty((bsz, n_q_heads, ns), device=dev, dtype=torch.float32)
    lp = torch.empty((bsz, n_q_heads, ns), device=dev, dtype=torch.float32)

    # strides for a [block, token, head, dim] view of the cache
    ks = k_cache.stride()
    vs = v_cache.stride()

    grid = (bsz * n_kv * ns,)
    _decode_split_kernel[grid](
        q, k_cache, v_cache, op, mp, lp, block_table, ctx_lens,
        q.stride(0), q.stride(1),
        ks[0], ks[1], ks[2],
        vs[0], vs[1], vs[2],
        op.stride(0), op.stride(1), op.stride(2),
        mp.stride(0), mp.stride(1),
        block_table.stride(0),
        scale,
        n_q_per_kv=n_per_kv, NUM_KV=n_kv, GROUP_PAD=group_pad, HEAD_DIM=head_dim,
        BLOCK_SIZE=blk, BLOCK_N=block_n, NUM_SPLITS=ns,
        num_warps=num_warps, num_stages=num_stages,
    )

    grid_m = (bsz,)
    _merge_kernel[grid_m](
        op, mp, lp, out,
        op.stride(0), op.stride(1), op.stride(2),
        mp.stride(0), mp.stride(1),
        out.stride(0), out.stride(1),
        n_q_heads=n_q_heads, GROUP_PAD=group_pad, HEAD_DIM=head_dim,
        NUM_SPLITS=ns, SPLIT_POW2=_next_pow2(ns), QH_POW2=_next_pow2(n_q_heads),
        num_warps=4, num_stages=1,
    )
    return out


def paged_decode_no_split(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    ctx_lens: torch.Tensor,
    *,
    num_kv_heads: int,
    block_size: int | None = None,
    sm_scale: float | None = None,
    block_n: int = 64,
    group_pad: int = 16,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Same math, one workgroup per (seq, kv head): the pathology baseline."""
    bsz, n_q_heads, head_dim = q.shape
    n_kv = int(num_kv_heads)
    n_per_kv = n_q_heads // n_kv
    blk = int(block_size) if block_size else int(k_cache.shape[1])
    scale = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(q)
    ks, vs = k_cache.stride(), v_cache.stride()
    _decode_nosplit_kernel[(bsz, n_kv)](
        q, k_cache, v_cache, out, block_table, ctx_lens,
        q.stride(0), q.stride(1), ks[0], ks[1], ks[2], vs[0], vs[1], vs[2],
        out.stride(0), out.stride(1), block_table.stride(0),
        scale,
        n_q_per_kv=n_per_kv, GROUP_PAD=group_pad, HEAD_DIM=head_dim,
        BLOCK_SIZE=blk, BLOCK_N=block_n,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
