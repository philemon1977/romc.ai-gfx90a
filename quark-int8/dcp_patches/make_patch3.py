#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 0003_gfx90a_attention_dcp.patch（DCP-C：attention 侧分片过滤 + LSE 合并）。

前置：必须先应用 0001（本补丁的基线是 0001 之后的 backend 文件，靠 lse 返回值）。
只改 v1/attention/backends/mla/rocm_aiter_mla_sparse.py（已挂载，无新文件）。
"""
import difflib, os, shutil, sys, py_compile

TREE = "/home/qiba/ai/recipes/patches/gfx90a/ct_w4a16_dsv41_n0918/tree"
REL = "v1/attention/backends/mla/rocm_aiter_mla_sparse.py"
HERE = os.path.dirname(os.path.abspath(__file__))
BASE1 = os.path.join(HERE, "work", "backend.py")          # 0001 的输出
WORK3 = os.path.join(HERE, "work", "backend_0003.py")

def sub(text, old, new, tag):
    n = text.count(old)
    if n != 1:
        sys.exit("ANCHOR FAIL [%s]: count=%d" % (tag, n))
    return text.replace(old, new)

src = open(BASE1, encoding="utf8").read()
if "_triton_lse_buf" not in src:
    sys.exit("基线不对：work/backend.py 不是 0001 之后的版本（缺 _triton_lse_buf）")

src = sub(src, """    supports_dcp = False""", """    supports_dcp = True
    # DCP 需要每 token 的 softmax LSE；**合并由层完成**（mla_attention.py 里
    # `attn_out = self.dcp_manager.combine(attn_out, lse, ...)`），后端只负责把
    # (out, lse) 返回出去 —— 后端若自己再合并一次，就是双重合并、静默算错。
    can_return_lse_for_decode = True
    # 我们的内核写的是**自然对数底** LSE（m_final + log(l_final)）；基类注释明确警告
    # 底数写错会静默污染跨分片 softmax 分母，故显式声明而不依赖默认值。
    lse_base_on_e = True""", "dcp-capability-flags")

src = sub(src, """        (self.q_concat_buffer,) = current_workspace_manager().get_simultaneous(
            (q_concat_shape, vllm_config.model_config.dtype),
        )
""", """        (self.q_concat_buffer,) = current_workspace_manager().get_simultaneous(
            (q_concat_shape, vllm_config.model_config.dtype),
        )

        # ---- DCP（0003）：本 rank 只持有 1/N 的 latent KV 分片 ----
        # 注意：合并**不在这里做** —— 层 mla_attention.py 拿到 (out, lse) 后调
        # dcp_manager.combine(...) 完成跨分片合并；后端再做一次就是双重合并。
        parallel_config = vllm_config.parallel_config
        self.dcp_world_size = int(parallel_config.decode_context_parallel_size)
        self.cp_interleave = int(parallel_config.cp_kv_cache_interleave_size)
        if self.dcp_world_size > 1:
            from vllm.distributed import get_dcp_group

            self.dcp_rank = get_dcp_group().rank_in_group
        else:
            self.dcp_rank = 0
        # GLM-5.3 无 sinks（config 无键、deepseek_v2.py 全文无 sinks）；其它模型若
        # 同时开 sinks 与 DCP，本 rank 折一次 sink 会让全局分母多算 ⇒ 直接拒绝，
        # 别让它静默算错（正确做法是分片内核不折 sink、合并后再折，见 DCP_A_NOTES.md）。
        assert not (self.dcp_world_size > 1 and self.sinks is not None), (
            "DCP + attention sinks is not implemented: folding the sink per shard "
            "double-counts it in the merged softmax denominator."
        )
""", "init-dcp-fields")

src = sub(src, """        triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            attn_metadata.paged_kv_indptr,
            attn_metadata.paged_kv_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
        )
""", """        if self.dcp_world_size > 1:
            # DCP：indexer 给的是**全局** token id（各 rank 本地 top-K 已在
            # sparse_attn_indexer 侧合并成全局 top-K），这里只保留本 rank 拥有的、
            # 并换算成本 rank 的物理槽位；compaction 把有效项压到行首，行尾留 -1。
            # 行长度契约（paged_kv_indptr，按 min(ctx, topk) 生成）不变：
            # 行内多出来的位置就是 -1 空洞，稀疏内核本来就掩 slot < 0。
            from vllm.v1.attention.backends.mla.sparse_utils import (
                triton_filter_and_convert_dcp_index,
            )

            local_slots, _valid_counts = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token,
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=self.cp_interleave,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
                compact_valid_to_front=True,
                return_valid_counts=True,
            )
            fetch_id_to_ragged_triton(
                local_slots,
                attn_metadata.paged_kv_indptr,
                attn_metadata.paged_kv_indices,
                attn_metadata.topk_tokens,
            )
        else:
            triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token,
                attn_metadata.block_table,
                topk_indices,
                attn_metadata.paged_kv_indptr,
                attn_metadata.paged_kv_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
            )
""", "forward-convert-dispatch")


open(WORK3, "w", encoding="utf8").write(src)
orig1 = open(BASE1, encoding="utf8").read()
patch = "".join(difflib.unified_diff(
    orig1.splitlines(keepends=True), src.splitlines(keepends=True),
    fromfile="a/" + REL, tofile="b/" + REL, n=3))
hdr = ("# DCP-C: attention-side DCP (per-rank slot filtering + fp32 LSE merge).\n"
       "# REQUIRES 0001 applied first (baseline = post-0001 backend file).\n"
       "# APPLY: cd " + TREE + " && patch -p1 < PATCH\n# REVERT: patch -p1 -R < PATCH\n")
out = os.path.join(HERE, "0003_gfx90a_attention_dcp.patch")
open(out, "w", encoding="utf8").write(hdr + patch)
cc = os.path.join(HERE, "work", "backend_0003.compilecheck.py")
shutil.copyfile(WORK3, cc)
py_compile.compile(cc, doraise=True)
print("OK patch=%s hunks=%d lines=%d" % (out, sum(1 for l in patch.splitlines() if l.startswith("@@ ")), len(patch.splitlines())))
