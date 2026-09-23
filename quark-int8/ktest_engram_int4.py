#!/usr/bin/env python3
"""_engram_lookup_kernel INT4 分支单测：Triton 实现 vs numpy 独立参考。

为什么值得单独测：nibble 顺序 / 两补码符号 / scale 字节语义任一处错了，
模型不会崩，只会**静默出垃圾**。这里用 256 列、4 个 head 的小表直接对拍。

用法（容器内，需挂载补丁后的 engram.py）:
  python3 ktest_engram_int4.py
"""
import numpy as np
import torch

from vllm.models.deepseek_v4_1.common.engram import _engram_lookup_kernel

DIM = 256
BS = 32
PART_ROWS = 37
T = 5
TOTAL_HEADS = 4
LOCAL_HEADS = 2
HEAD_START = 0
VOCAB_START = 1000
BLOCK_R = 16
DEV = "cuda"


def ref(weight, scales, ids, vocab_start, vocab_end):
    """numpy 参考：只读本 rank 的 head，nibble→两补码→×2^(s-127)。"""
    out = np.zeros((T * LOCAL_HEADS, DIM), dtype=np.float32)
    for r in range(T * LOCAL_HEADS):
        tok = r // LOCAL_HEADS
        h = HEAD_START + r % LOCAL_HEADS
        if h >= TOTAL_HEADS:
            continue
        idx = int(ids[tok, h])
        if not (vocab_start <= idx < vocab_end):
            continue
        loc = idx - vocab_start
        if loc >= weight.shape[0]:
            continue
        packed = weight[loc]                       # [128] uint8
        codes = np.empty(DIM, dtype=np.int32)
        codes[0::2] = packed & 0x0F
        codes[1::2] = (packed >> 4) & 0x0F
        codes = np.where(codes >= 8, codes - 16, codes).astype(np.float32)
        sc = np.exp2(scales[loc].astype(np.int32) - 127).astype(np.float32)
        out[r] = codes * np.repeat(sc, BS)
    return out


def main():
    rng = np.random.default_rng(0)
    w = rng.integers(0, 256, size=(PART_ROWS, DIM // 2), dtype=np.uint8)
    s = rng.integers(120, 132, size=(PART_ROWS, DIM // BS), dtype=np.uint8)
    ids = rng.integers(VOCAB_START, VOCAB_START + PART_ROWS, size=(T, TOTAL_HEADS),
                       dtype=np.int64)
    ids[:, LOCAL_HEADS:] = -1                      # 非本 rank 的 head

    tw = torch.from_numpy(w).to(DEV)
    ts = torch.from_numpy(s).to(DEV)
    ti = torch.from_numpy(ids).to(DEV)
    out = torch.zeros((T * LOCAL_HEADS, DIM), dtype=torch.bfloat16, device=DEV)
    rows = T * LOCAL_HEADS
    grid = 4
    _engram_lookup_kernel[(grid,)](
        tw, ts, ti, out, VOCAB_START, VOCAB_START + PART_ROWS, rows,
        ti.stride(0), ti.stride(1),
        HEAD_START=HEAD_START, LOCAL_HEADS=LOCAL_HEADS, TOTAL_HEADS=TOTAL_HEADS,
        DIM=DIM, QUANT_BLOCK=BS, INT4=True, BLOCK_R=BLOCK_R, GRID=grid,
    )
    got = out.float().cpu().numpy()
    exp = ref(w, s, ids, VOCAB_START, VOCAB_START + PART_ROWS)
    d = np.abs(got - exp)
    rel = d / np.maximum(np.abs(exp), 1e-30)
    print(f"INT4=True  最大绝对误差 {d.max():.6g}   最大相对误差 {rel.max():.6g}")
    ok = d.max() < 1e-3
    # 反向门：INT4=False 必须**对不上**（否则说明 INT4 分支没生效）
    out2 = torch.zeros_like(out)
    _engram_lookup_kernel[(grid,)](
        tw, ts, ti, out2, VOCAB_START, VOCAB_START + PART_ROWS, rows,
        ti.stride(0), ti.stride(1),
        HEAD_START=HEAD_START, LOCAL_HEADS=LOCAL_HEADS, TOTAL_HEADS=TOTAL_HEADS,
        DIM=DIM, QUANT_BLOCK=BS, INT4=False, BLOCK_R=BLOCK_R, GRID=grid,
    )
    d2 = np.abs(out2.float().cpu().numpy() - exp).max()
    print(f"INT4=False 最大绝对误差 {d2:.6g}（应显著 >0，证明分支真的分了）")
    if ok and d2 > 1e-2:
        print("[PASS] nibble 序 / 两补码 / ue8m0 scale 语义与 numpy 参考一致")
    else:
        print("[FAIL] INT4 分支与参考不符")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
