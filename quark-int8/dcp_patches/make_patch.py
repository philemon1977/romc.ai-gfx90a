#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 0001_gfx90a_sparse_mla_lse.patch（DCP 改造 A：让 Triton 稀疏分支写 LSE）。

- 只动生产链上的单发 prefill-ragged 内核；split-KV decode 内核此链未用，留作后手。
- WRITE_LSE constexpr：不传 lse 时内核行为与今天逐比特一致（零风险默认路径）。
- 后端 _forward_mla Triton 分支复用缓冲（cudagraph 安全），返回真实 lse。
本脚本只生成 patch + 语法门，**不把补丁打到真树**；⓪ 针尖门过了才允许打。
"""
import difflib, os, shutil, sys, py_compile

TREE = "/home/qiba/ai/recipes/patches/gfx90a/ct_w4a16_dsv41_n0918/tree"
OPS_REL = "v1/attention/ops/rocm_aiter_mla_sparse.py"
BE_REL = "v1/attention/backends/mla/rocm_aiter_mla_sparse.py"
HERE = os.path.dirname(os.path.abspath(__file__))
BASE, WORK = os.path.join(HERE, "base"), os.path.join(HERE, "work")

def sub(text, old, new, tag):
    n = text.count(old)
    if n != 1:
        sys.exit("ANCHOR FAIL [%s]: count=%d" % (tag, n))
    return text.replace(old, new)

os.makedirs(BASE, exist_ok=True); os.makedirs(WORK, exist_ok=True)

ops = open(os.path.join(TREE, OPS_REL), encoding="utf8").read()

ops = sub(ops, """    out_stride_d,
    num_heads,
    head_dim,
    num_kv,
    scale,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):""", """    out_stride_d,
    lse_ptr,
    lse_stride_t,
    lse_stride_h,
    num_heads,
    head_dim,
    num_kv,
    scale,
    HAS_ATTN_SINK: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):""", "ops-kernel-signature")

ops = sub(ops, """    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)

    tl.store(""", """    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)
        if WRITE_LSE:
            lse = tl.where(l_i > 0.0, m_i + tl.log(l_i), float("-inf"))

    # DCP-A：每 (query, head) 一个 ln 底 LSE；空行 -inf（sink 分支在上方各自折叠）。
    if WRITE_LSE:
        tl.store(
            lse_ptr + query_idx * lse_stride_t + head_offsets * lse_stride_h,
            lse,
            mask=head_mask,
        )

    tl.store(""", "ops-kernel-epilogue-else")

ops = sub(ops, """        out = tl.where(
            l_final[:, None] > 0.0,
            (acc * alpha[:, None]) / denom[:, None],
            0.0,
        )
""", """        out = tl.where(
            l_final[:, None] > 0.0,
            (acc * alpha[:, None]) / denom[:, None],
            0.0,
        )
        if WRITE_LSE:
            lse = tl.where(l_final > 0.0, m_final + tl.log(l_final), float("-inf"))
""", "ops-kernel-epilogue-sink")

ops = sub(ops, """    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[sq,h,d], got {q.shape}"
    assert kv.ndim == 2, f"expected kv=[skv,d], got {kv.shape}"
""", """    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    lse: torch.Tensor | None = None,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[sq,h,d], got {q.shape}"
    assert kv.ndim == 2, f"expected kv=[skv,d], got {kv.shape}"
""", "ops-inner-wrapper-sig")

ops = sub(ops, """    out = torch.empty_like(q)
    _sparse_attn_prefill_ragged_kernel[(num_queries, triton.cdiv(num_heads, block_h))](""", """    out = torch.empty_like(q)
    write_lse = lse is not None
    if write_lse:
        assert (
            lse.dtype == torch.float32
            and lse.shape == (num_queries, num_heads)
            and lse.is_contiguous()
        ), "lse must be contiguous fp32 [%d,%d]" % (num_queries, num_heads)
        lse_arg = lse
    else:
        lse_arg = out  # WRITE_LSE=False 时不会被解引用，占位即可
    _sparse_attn_prefill_ragged_kernel[(num_queries, triton.cdiv(num_heads, block_h))](""", "ops-inner-wrapper-body")

ops = sub(ops, """        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        head_dim,
        kv.shape[0],
        float(scale),
        HAS_ATTN_SINK=has_attn_sink,
        BLOCK_H=block_h,""", """        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse_arg,
        lse_arg.stride(0),
        lse_arg.stride(1),
        num_heads,
        head_dim,
        kv.shape[0],
        float(scale),
        HAS_ATTN_SINK=has_attn_sink,
        WRITE_LSE=write_lse,
        BLOCK_H=block_h,""", "ops-inner-launch")

ops = sub(ops, """    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
    ragged_indices: torch.Tensor | None = None,
    ragged_indptr: torch.Tensor | None = None,
) -> None:""", """    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
    ragged_indices: torch.Tensor | None = None,
    ragged_indptr: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
) -> None:""", "ops-public-sig")

ops = sub(ops, """        if _rocm_sparse_attn_prefill_ragged_aiter_opus(""", """        if output_lse is not None:
            raise NotImplementedError(
                "AITER opus sparse prefill does not produce LSE; DCP needs "
                "the Triton ragged path (do not route DCP here)."
            )
        if _rocm_sparse_attn_prefill_ragged_aiter_opus(""", "ops-public-opus-guard")

ops = sub(ops, """            attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
        )
    else:
        assert indices is not None""", """            attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
            lse=output_lse,
        )
    else:
        if output_lse is not None:
            raise NotImplementedError(
                "LSE is only produced by the ragged Triton sparse prefill; "
                "DCP must pass ragged_indices/ragged_indptr."
            )
        assert indices is not None""", "ops-public-dispatch")

be = open(os.path.join(TREE, BE_REL), encoding="utf8").read()

be = sub(be, """            rocm_sparse_attn_prefill(
                q=q,
                kv=kv_c_and_k_pe_cache.view(-1, 1, q.shape[-1]),
                indices=None,
                topk_length=None,
                scale=self.scale,
                head_dim=q.shape[-1],
                nope_head_dim=self.kv_lora_rank,
                rope_head_dim=q.shape[-1] - self.kv_lora_rank,
                attn_sink=triton_sinks,
                output=output,
                ragged_indices=attn_metadata.paged_kv_indices,
                ragged_indptr=attn_metadata.paged_kv_indptr,
            )
            output = AiterMLAHelper.get_mla_unpadded_o(self.num_heads, output)
            return output, None""", """            # DCP 改造 A：Triton 分支产出每 (token, head) 的 LSE（ln 底，sink 已折叠）。
            # 缓冲复用（cudagraph 安全）；返回契约与 AITER 分支一致（第二元是 LSE）。
            buf = getattr(self, "_triton_lse_buf", None)
            if (
                buf is None
                or buf.shape[0] < num_tokens
                or buf.shape[1] != q.shape[1]
                or buf.device != q.device
            ):
                buf = torch.empty(
                    (max(num_tokens, 512), q.shape[1]),
                    dtype=torch.float32,
                    device=q.device,
                )
                self._triton_lse_buf = buf
            lse = buf[:num_tokens]
            rocm_sparse_attn_prefill(
                q=q,
                kv=kv_c_and_k_pe_cache.view(-1, 1, q.shape[-1]),
                indices=None,
                topk_length=None,
                scale=self.scale,
                head_dim=q.shape[-1],
                nope_head_dim=self.kv_lora_rank,
                rope_head_dim=q.shape[-1] - self.kv_lora_rank,
                attn_sink=triton_sinks,
                output=output,
                ragged_indices=attn_metadata.paged_kv_indices,
                ragged_indptr=attn_metadata.paged_kv_indptr,
                output_lse=lse,
            )
            output = AiterMLAHelper.get_mla_unpadded_o(self.num_heads, output)
            lse = AiterMLAHelper.get_mla_unpadded_o(
                self.num_heads, lse.unsqueeze(-1)
            ).squeeze(-1)
            return output, lse""", "backend-forward-triton")

ops_orig = open(os.path.join(TREE, OPS_REL), encoding="utf8").read()
be_orig = open(os.path.join(TREE, BE_REL), encoding="utf8").read()
shutil.copyfile(os.path.join(TREE, OPS_REL), os.path.join(BASE, "ops.py.orig"))
shutil.copyfile(os.path.join(TREE, BE_REL), os.path.join(BASE, "backend.py.orig"))
open(os.path.join(WORK, "ops.py"), "w", encoding="utf8").write(ops)
open(os.path.join(WORK, "backend.py"), "w", encoding="utf8").write(be)

def diff(rel, old, new):
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile="a/" + rel, tofile="b/" + rel, n=3))

patch = (
    "# DCP-A: gfx90a Triton sparse MLA writes LSE (per token x head, ln base, empty=-inf).\n"
    "# APPLY: cd " + TREE + " && patch -p1 --dry-run < PATCH   (dry-run first; real apply only after gate 0)\n"
    "# REVERT: patch -p1 -R < PATCH\n"
    + diff(OPS_REL, ops_orig, ops) + diff(BE_REL, be_orig, be))
out = os.path.join(HERE, "0001_gfx90a_sparse_mla_lse.patch")
open(out, "w", encoding="utf8").write(patch)

for name in ("ops", "backend"):
    f = os.path.join(WORK, name + ".compilecheck.py")
    shutil.copyfile(os.path.join(WORK, name + ".py"), f)
    py_compile.compile(f, doraise=True)
hunks = sum(1 for ln in patch.splitlines() if ln.startswith("@@ "))
print("OK patch=%s hunks=%d lines=%d" % (out, hunks, len(patch.splitlines())))
