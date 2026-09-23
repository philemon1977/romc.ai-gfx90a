#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 0005（ROCm indexer 的 DCP top-K 合并）与 0006（逐行长度本地化）。
依据 quark-int8/DCP_A_NOTES.md 的两条取证结论（缺陷①/②）。只改已挂载文件。"""
import difflib, os, shutil, sys, py_compile

TREE = "/home/qiba/ai/recipes/patches/gfx90a/ct_w4a16_dsv41_n0918/tree"
OPS_REL = "v1/attention/ops/rocm_aiter_mla_sparse.py"
BE_REL = "v1/attention/backends/mla/rocm_aiter_mla_sparse.py"
HERE = os.path.dirname(os.path.abspath(__file__))
BASE, WORK = os.path.join(HERE, "base"), os.path.join(HERE, "work")
os.makedirs(BASE, exist_ok=True); os.makedirs(WORK, exist_ok=True)

def sub(text, old, new, tag):
    n = text.count(old)
    if n != 1:
        sys.exit("ANCHOR FAIL [%s]: count=%d" % (tag, n))
    return text.replace(old, new)

def emit(name, rel, old, new):
    patch = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                    fromfile="a/" + rel, tofile="b/" + rel, n=3))
    hdr = ("# " + name + "  (see quark-int8/DCP_A_NOTES.md)" + chr(10) +
           "# APPLY: cd " + TREE + " && patch -p1 < PATCH   (dry-run first)" + chr(10) +
           "# REVERT: patch -p1 -R < PATCH" + chr(10))
    out = os.path.join(HERE, name)
    open(out, "w", encoding="utf8").write(hdr + patch)
    print("wrote %s hunks=%d lines=%d" % (out, sum(1 for l in patch.splitlines() if l.startswith("@@ ")), len(patch.splitlines())))

# ---------------- 0005 ----------------
ops_path = os.path.join(TREE, OPS_REL)
ops0 = open(ops_path, encoding="utf8").read(); ops = ops0

HELPER = chr(39)*3 + "placeholder" + chr(39)*3
helper_code = """def _dcp_merge_topk_if_needed(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    row_starts: torch.Tensor | None = None,
) -> None:
    # DCP 缺陷①（2026-09-20 取证，见 DCP_A_NOTES.md）：ROCm 走
    # rocm_aiter_sparse_attn_indexer，不经过 CUDA 侧 sparse_attn_indexer 里的
    # _merge_dcp_topk_global；而 DCP 下 index-K 也是分片的，所以这里选出的 top-K
    # 是**本 rank 的本地位置**。注意力侧按全局 id 过滤 ⇒ 除 rank0 外有效槽全为 0
    # ⇒ 内核读到空（out=0、lse=-inf）⇒ 输出确定性乱码。
    # dcp 参数就地取，不改 op 签名（改签名要连带动 fake/注册/调用方三处）。
    from vllm.config import get_current_vllm_config

    parallel_config = get_current_vllm_config().parallel_config
    dcp_world_size = int(parallel_config.decode_context_parallel_size)
    if dcp_world_size <= 1:
        return

    from vllm.distributed import get_dcp_group
    # 复用 0002 已写好并通过审查的实现（Triton pack + torch stable-topk）。
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _merge_dcp_topk_global_gfx90a,
    )

    _merge_dcp_topk_global_gfx90a(
        logits,
        topk_indices,
        topk_tokens,
        get_dcp_group().rank_in_group,
        dcp_world_size,
        int(parallel_config.cp_kv_cache_interleave_size),
        row_starts,
    )


def rocm_aiter_sparse_attn_indexer_fake("""
ops = sub(ops, "def rocm_aiter_sparse_attn_indexer_fake(", helper_code, "ops-helper")

old_p = """                    logits.stride(1),
                    topk_tokens,
                )

            if os.environ.get("DSV41_IDX_DUMP", "0") == "1\""""
new_p = """                    logits.stride(1),
                    topk_tokens,
                )

            # DCP 缺陷①：本地 top-K -> 全局 token id。prefill 的行在打包 logits 里
            # 从 chunk.cu_seqlen_ks 起，故要传 row_starts。
            _dcp_merge_topk_if_needed(
                logits, topk_indices, topk_tokens, row_starts=chunk.cu_seqlen_ks
            )

            if os.environ.get("DSV41_IDX_DUMP", "0") == "1\""""
ops = sub(ops, old_p, new_p, "ops-prefill-merge")

old_d = """                topk_tokens,
            )

        global _DSV41_IDX_DBG_DONE"""
new_d = """                topk_tokens,
            )

        # DCP 缺陷①：decode 侧同样要换成全局 id（logits 行从 0 起 => 无 row_starts）。
        _dcp_merge_topk_if_needed(logits, topk_indices, topk_tokens)

        global _DSV41_IDX_DBG_DONE"""
ops = sub(ops, old_d, new_d, "ops-decode-merge")

emit("0005_gfx90a_rocm_indexer_dcp_topk.patch", OPS_REL, ops0, ops)
shutil.copyfile(ops_path, os.path.join(BASE, "ops.pre0005.py"))
open(os.path.join(WORK, "ops_0005.py"), "w", encoding="utf8").write(ops)

# ---------------- 0006 ----------------
be_path = os.path.join(TREE, BE_REL)
be0 = open(be_path, encoding="utf8").read(); be = be0

kern = '''@triton.jit
def _generate_sparse_seqlen_dcp_kernel(
    seq_len_ptr,  # [num_seq] 全局上下文长度
    cu_query_lens_ptr,  # [num_seq]
    out_ptr,  # [num_query_tokens] 本 rank 可见长度（已 clamp 到 topk）
    topk_token: tl.constexpr,
    DCP_WORLD: tl.constexpr,
    DCP_RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # DCP 缺陷②（2026-09-20 取证）：原写法把"总长"先本地化再减 query_len，
    # 当本 rank 分到的 token 数小于 query_len 时（3-token prompt、dcp=8 => rank0
    # 本地长度 1）context_start = 1-3 = -2 => 行长度为负、paged_kv_indptr 非单调，
    # 还会把有效槽截掉（实测 valid>len）。正确做法：先算每行的全局前缀长度
    # prefix = (seq_len - query_len) + offset + 1，再按 get_dcp_local_seq_lens 同一公式本地化。
    seq_id = tl.program_id(0)
    query_offset = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    query_start = tl.load(cu_query_lens_ptr + seq_id)
    query_end = tl.load(cu_query_lens_ptr + seq_id + 1)
    if query_start + tl.program_id(1) * BLOCK_SIZE > query_end:
        return
    query_len = query_end - query_start
    query_mask = query_offset + query_start < query_end
    seq_len = tl.load(seq_len_ptr + seq_id)
    if seq_len == 0:
        return
    prefix = (seq_len - query_len) + query_offset + 1

    base = (prefix // INTERLEAVE // DCP_WORLD) * INTERLEAVE
    remainder = prefix - base * DCP_WORLD - DCP_RANK * INTERLEAVE
    remainder = tl.minimum(tl.maximum(remainder, 0), INTERLEAVE)
    local = base + remainder

    out = tl.minimum(local, topk_token)
    tl.store(out_ptr + query_start + query_offset, out.to(tl.int32), mask=query_mask)


def generate_sparse_seqlen_dcp_triton(
    query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_query_lens: torch.Tensor,
    topk_token: int,
    num_tokens: int,
    max_query_len: int,
    dcp_world: int,
    dcp_rank: int,
    interleave: int,
) -> torch.Tensor:
    """generate_sparse_seqlen_triton 的 DCP 版（见上方内核说明）。"""
    num_seqs = query_lens.size(0)
    out = torch.zeros([num_tokens], dtype=torch.int32, device=query_lens.device)
    block_size = 64
    grid = (num_seqs, triton.cdiv(max_query_len, block_size))
    _generate_sparse_seqlen_dcp_kernel[grid](
        seq_lens,
        cu_query_lens,
        out,
        topk_token,
        DCP_WORLD=dcp_world,
        DCP_RANK=dcp_rank,
        INTERLEAVE=interleave,
        BLOCK_SIZE=block_size,
    )
    return out


def fetch_id_to_ragged_kernel('''
be = sub(be, "def fetch_id_to_ragged_kernel(", kern, "be-kernel")

old_b = """        seq_lens = common_attn_metadata.seq_lens
        if self.dcp_world_size > 1:
            # 上游公式（与 indexer 的 Triton 侧逐位一致）：interleave 感知的本 rank 分片长度。
            from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

            seq_lens = get_dcp_local_seq_lens(
                seq_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_interleave,
            )
        sparse_seqlen = generate_sparse_seqlen_triton(
            query_lens,
            seq_lens,
            common_attn_metadata.query_start_loc,
            self.topk_tokens,
            num_tokens,
            common_attn_metadata.max_query_len,
        )
"""
new_b = """        seq_lens = common_attn_metadata.seq_lens
        # 层的 dcp_manager.combine 需要**本地**分片长度（空分片权重 0）。
        local_seq_lens = seq_lens
        if self.dcp_world_size > 1:
            from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

            local_seq_lens = get_dcp_local_seq_lens(
                seq_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_interleave,
            )
        if self.dcp_world_size > 1:
            # 缺陷②：行长度按"每行自己的全局前缀长度"本地化，不能拿本地总长减 query_len。
            sparse_seqlen = generate_sparse_seqlen_dcp_triton(
                query_lens,
                seq_lens,
                common_attn_metadata.query_start_loc,
                self.topk_tokens,
                num_tokens,
                common_attn_metadata.max_query_len,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_interleave,
            )
        else:
            sparse_seqlen = generate_sparse_seqlen_triton(
                query_lens,
                seq_lens,
                common_attn_metadata.query_start_loc,
                self.topk_tokens,
                num_tokens,
                common_attn_metadata.max_query_len,
            )
"""
be = sub(be, old_b, new_b, "be-build")
be = sub(be, "            seq_lens=seq_lens,\n            slot_mapping=common_attn_metadata.slot_mapping,",
         "            seq_lens=local_seq_lens,\n            slot_mapping=common_attn_metadata.slot_mapping,", "be-meta")

emit("0006_gfx90a_dcp_row_local_lengths.patch", BE_REL, be0, be)
shutil.copyfile(be_path, os.path.join(BASE, "backend.pre0006.py"))
open(os.path.join(WORK, "backend_0006.py"), "w", encoding="utf8").write(be)

for nm, txt in (("ops_0005", ops), ("backend_0006", be)):
    f = os.path.join(WORK, nm + ".compile.py")
    open(f, "w", encoding="utf8").write(txt)
    py_compile.compile(f, doraise=True)
print("COMPILE_OK both")
