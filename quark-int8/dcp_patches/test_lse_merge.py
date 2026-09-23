#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCP-A 尺子2+3（单进程单卡即可）：LSE 正确性 & 分片合并等价。
跑法（空闲卡窗口，勿与 8 卡服务抢卡）：
  docker run --rm --entrypoint python3 -v /home/qiba/ROCm.AI:/workspace \
    -e HIP_VISIBLE_DEVICES=<空闲卡> vllm/vllm-openai-rocm:nightly-0918 \
    /workspace/quark-int8/dcp_patches/test_lse_merge.py
前提：0001 补丁已在生效链路上（真树挂载起服同样满足）。
维度用生产同款：nope=512, rope=64, head=576。
"""
import math
import torch

from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _rocm_sparse_attn_prefill_ragged_triton as ragged,
)

DEV = "cuda"
DT = torch.bfloat16
NOPE, ROPE = 512, 64
HD = NOPE + ROPE
H = 8          # 局部 heads（64/TP8 与生产一致）
SCALE = 1.0 / math.sqrt(HD)


def ref_out_lse(q, kv, rows, sink=None):
    """q [M,H,HD] bf16; rows: list of slot lists (含 -1 尾部)。返回 fp32。"""
    k = kv.float()
    outs, lses = [], []
    for m in range(q.shape[0]):
        slots = [s for s in rows[m] if s >= 0]
        s = q[m].float() @ k[slots].T * SCALE          # [H, n]
        mx = s.max(dim=1).values                        # [H]
        e = torch.exp(s - mx[:, None])                  # [H, n]
        l = e.sum(1)
        if sink is not None:
            mx2 = torch.maximum(mx, sink)
            e = e * torch.exp(mx - mx2)[:, None]
            l = l * torch.exp(mx - mx2) + torch.exp(sink - mx2)
            mx = mx2
        out = (e @ k[slots]) / l[:, None]
        outs.append(out)
        lses.append(mx + torch.log(l))
    return torch.stack(outs), torch.stack(lses)


def main():
    torch.manual_seed(7)
    M = 5
    kv_rows = 96
    q = (torch.randn(M, H, HD, device=DEV, dtype=DT) * 0.5)
    kv = (torch.randn(kv_rows, HD, device=DEV, dtype=DT) * 0.5)
    # 造 ragged：行0 全量40；行1 25个有效(-1尾部)；行2 空行(全-1，本rank无持有)；行3/4 半量乱序。
    lens = [40, 25, 0, 18, 33]
    rows = []
    for i, ln in enumerate(lens):
        sel = torch.randperm(kv_rows, device=DEV)[:ln].tolist()
        rows.append(sel + [-1] * (40 - ln))
    flat = torch.tensor([s for r in rows for s in r], dtype=torch.int32, device=DEV)
    indptr = torch.tensor([40 * i for i in range(M + 1)], dtype=torch.int32, device=DEV)

    # 尺子2a：基线 out vs torch 参考；并留作尺子3的全量答案
    out0 = ragged(q, kv, flat, indptr, SCALE, None, NOPE, ROPE)
    o_ref, l_ref = ref_out_lse(q, kv, rows)
    err = (out0.float() - o_ref).abs().max().item()
    assert err < 5e-2, f"out mismatch {err}"
    print(f"[ruler2a] out ok, maxerr={err:.2e}")

    # 尺子2b：lse == masked logsumexp（ln 底）；空行 -inf；out 逐比特不变
    lse = torch.zeros(M, H, dtype=torch.float32, device=DEV)
    out1 = ragged(q, kv, flat, indptr, SCALE, None, NOPE, ROPE, lse=lse)
    assert torch.equal(out0, out1), "WRITE_LSE must not change out"
    for m in range(M):
        if lens[m] == 0:
            assert torch.all(lse[m] == float("-inf")), f"empty row lse {lse[m]}"
            continue
        err = (lse[m] - l_ref[m]).abs().max().item()
        rel = err / l_ref[m].abs().max().clamp(min=1).item()
        assert rel < 2e-3, f"row{m} lse err {err} rel {rel}"
    print("[ruler2b] lse ok (incl. empty-row -inf), out bitwise unchanged")

    # 尺子2c：sink 折叠语义；空行+sink 时 lse==sink（只剩 sink 质量）
    sink = torch.randn(H, device=DEV) * 0.3
    lse_s = torch.zeros(M, H, dtype=torch.float32, device=DEV)
    ragged(q, kv, flat, indptr, SCALE, sink, NOPE, ROPE, lse=lse_s)
    _, l_ref_s = ref_out_lse(q, kv, rows, sink=sink)
    for m in range(M):
        if lens[m] == 0:
            assert torch.allclose(lse_s[m], sink, atol=1e-3), f"empty+sink row{m}"
            continue
        assert (lse_s[m] - l_ref_s[m]).abs().max().item() < 5e-3
    print("[ruler2c] sink-folded lse ok")

    # 尺子3：行拆两 shard，LSE 加权合并 == 全量（模拟 DCP 合并公式）
    for m in [0, 1, 3, 4]:
        sel = [s for s in rows[m] if s >= 0]
        half = len(sel) // 2
        shardA = sel[:half] + [-1] * (40 - half)
        rest = sel[half:]
        shardB = rest + [-1] * (40 - len(rest))
        qm = q[m : m + 1]

        def run(shard):
            f = torch.tensor(shard, dtype=torch.int32, device=DEV)
            ip = torch.tensor([0, len(shard)], dtype=torch.int32, device=DEV)
            L = torch.zeros(1, H, dtype=torch.float32, device=DEV)
            O = ragged(qm, kv, f, ip, SCALE, None, NOPE, ROPE, lse=L)
            return O, L

        OA, LA = run(shardA)
        if len(rest) == 0:
            OB = torch.zeros_like(OA)
            LB = torch.full_like(LA, float("-inf"))
        else:
            OB, LB = run(shardB)
        mx = torch.maximum(LA, LB)
        wa = torch.exp(LA - mx)
        wb = torch.exp(LB - mx)
        merged = (wa.unsqueeze(-1) * OA.float() + wb.unsqueeze(-1) * OB.float()) / (
            wa + wb
        ).unsqueeze(-1)
        full = out0[m : m + 1].float()
        err = (merged - full).abs().max().item()
        assert err < 2e-2, f"merge row{m} err {err}"
        print(f"[ruler3] row{m} merge ok maxerr={err:.2e}")

    print("ALL LSE/MERGE RULERS PASS")


if __name__ == "__main__":
    main()
