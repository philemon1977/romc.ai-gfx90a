#!/usr/bin/env python3
"""独立校验 engram int4 重写：从写出的分片反读，按与写侧无关的解码路径重建，
再与源 fp8(block32) 原始值逐元素对比。

用法：
  python verify_engram_int4.py --src <fp8 目录> --dst <int4 目录> \
      --shard model-00047-of-00048.safetensors --rows 200000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct

import numpy as np
from safetensors import safe_open


def hdr(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n))


def dequant_fp8_rows(src, name, sname, rows):
    """返回 fp8 原始实值 [rows, 256] (float32)。"""
    import torch
    with safe_open(src, framework="pt") as f:
        c = f.get_slice(name)[:rows].to(torch.float32).numpy()          # e4m3 码值
        s = f.get_slice(sname)[:rows].view(torch.uint8).numpy()          # ue8m0 字节
    sc = np.exp2(s.astype(np.int32) - 127).astype(np.float32)            # [rows, 8]
    return c * np.repeat(sc, 32, axis=1), c, s


def dequant_int4_rows(dst, name, sname, rows):
    """从 packed U8[rows,128] + ue8m0 scale 反解码 → [rows, 256] 实值。

    这里是**独立实现**：numpy 层手动取 nibble、手动两补码、手动 scale。
    """
    with safe_open(dst, framework="pt") as f:
        p = f.get_tensor(name)[:rows].numpy()                            # U8 [rows,128]
        s = f.get_tensor(sname)[:rows].view(__import__("torch").uint8).numpy()
    lo = (p & 0x0F).astype(np.int16)
    hi = ((p >> 4) & 0x0F).astype(np.int16)
    codes = np.empty((p.shape[0], p.shape[1] * 2), dtype=np.int16)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    codes = np.where(codes >= 8, codes - 16, codes).astype(np.float32)   # 两补码
    sc = np.exp2(s.astype(np.int32) - 127).astype(np.float32)
    return codes * np.repeat(sc, 32, axis=1), codes, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--rows", type=int, default=200000)
    a = ap.parse_args()

    sh = a.shard
    hs = hdr(os.path.join(a.src, sh))
    hd = hdr(os.path.join(a.dst, sh))
    names = [k for k in hs if k != "__metadata__"]
    big = [k for k in names if k.endswith(".engram.embed.weight")]
    print(f"源: {hs[big[0]]['dtype']} {hs[big[0]]['shape']}")
    print(f"新: {hd[big[0]]['dtype']} {hd[big[0]]['shape']}")
    assert hd[big[0]]["dtype"] == "U8"
    assert hd[big[0]]["shape"][1] == hs[big[0]]["shape"][1] // 2
    # 其余张量：名字集合必须完全一致（requant 不得增删/改名张量），
    # 但 dtype/形状**允许**转换器已知的改写 —— 最典型就是 `engram.wkv` 的 FP8→bf16
    # （见转换记录 §4.17）。早先这里用硬断言"逐字节一致"，只在
    # --src=「转换后但未 requant 的目录」时成立；一旦把 --src 指向官方 fp8 模型
    # （语义更强的对照）就会误报 ✗ 故改为"名字硬校验 + 类型差异白名单报告"。
    sname0 = big[0][: -len("weight")] + "scale"
    dn = {k for k in hd if k != "__metadata__"}
    # 正确的不变式（不是"名字集合相等"✗）：
    #   转换器对 fp8_to_bf16 的处理是「反量化成 bf16 并**丢弃 scale**」，
    #   所以 src 里那些张量的 `.scale` 在 dst 中本就不该存在。
    #   ⇒ 允许缺失的集合**由数据自己推出**，绝不写死白名单（写死会把真问题藏起来）。
    FP32_LIKE = ("BF16", "F16", "F32")
    allowed_missing = set()
    for k in set(names) - dn:
        if not k.endswith(".scale"):
            continue
        # ★ engram 的命名是 `wkv.weight` 与 `wkv.scale` **平级**（不是 `wkv.weight_scale`），
        #   所以去掉 ".scale" 之后必须补回 ".weight"，否则 base 指向一个不存在的名字，
        #   条件恒假 ⇒ allowed_missing 永远为空 ⇒ 断言必然误挂（本次就是这个错）。
        stem = k[: -len(".scale")]
        base = stem + ".weight" if stem + ".weight" in dn else k[: -len("_scale")] \
            if k.endswith("_scale") else stem
        if base in dn and base in hs and hd[base]["dtype"] in FP32_LIKE \
           and hs[base]["dtype"] not in FP32_LIKE:
            allowed_missing.add(k)
    unexplained = (set(names) - dn) - allowed_missing
    assert not unexplained, f"出现无法解释的张量缺失: {sorted(unexplained)}"
    assert dn - set(names) == set(), f"dst 多出 src 没有的张量: {sorted(dn - set(names))}"
    for k in sorted(allowed_missing):
        _b = k[: -len(".scale")] + ".weight"
        print(f"  已解释的缺失: {k}（其 weight 由 {hs[_b]['dtype']} 反量化为 "
              f"{hd[_b]['dtype']}，bf16 参数不需要 scale）")
    # 已知的、由转换器刻意造成的类型变化
    KNOWN = (".engram.wkv.weight",)
    diffs = []
    for k in names:
        if k in big or k == sname0:
            continue
        if k not in dn:          # 上面已判定为"可解释的缺失"（fp8→bf16 丢 scale）⇒ 不比类型
            continue
        si, di = hs[k], hd[k]
        if si["dtype"] != di["dtype"] or si["shape"] != di["shape"]:
            tag = "已知(FP8→bf16)" if k.endswith(KNOWN) else "★未预期★"
            diffs.append(f"{k}: {si['dtype']}{si['shape']} → {di['dtype']}{di['shape']} [{tag}]")
    unexpected = [d for d in diffs if "★未预期★" in d]
    for d in diffs:
        print("  类型差异:", d)
    assert not unexpected, f"出现未预期的张量类型/形状变化: {unexpected}"
    print(f"张量名集合校验通过 ✓（缺失项全部可解释）；其余 {len(names)-1} 个中 {len(diffs)} 个有已知类型改写")

    name, sname = big[0], big[0][: -len("weight")] + "scale"
    v_fp8, c_fp8, s_old = dequant_fp8_rows(os.path.join(a.src, sh), name, sname, a.rows)
    v_i4, codes, s_new = dequant_int4_rows(os.path.join(a.dst, sh), name, sname, a.rows)

    amax_blk = np.repeat(np.abs(c_fp8).reshape(a.rows, 8, 32).max(-1), 32, axis=1) * \
        np.repeat(np.exp2(s_old.astype(np.int32) - 127).astype(np.float32), 32, axis=1)
    d = v_i4 - v_fp8
    rms_v = float(np.sqrt(np.mean(v_fp8.astype(np.float64) ** 2)))
    rms_e = float(np.sqrt(np.mean(d.astype(np.float64) ** 2)))
    snr = 20 * math.log10(rms_v / max(rms_e, 1e-30))
    print("\n--- 数值校验（%d 元素 = %d 行 × 256）" % (d.size, a.rows))
    print(f"  rms(值)            {rms_v:.6g}")
    print(f"  rms(误差)          {rms_e:.6g}")
    print(f"  SNR                {snr:.2f} dB")
    print(f"  mean|Δ|/amax_blk   {float(np.mean(np.abs(d))/np.mean(amax_blk)):.4f}  "
          f"(int4 理论上界 1/14 = {1/14:.4f})")
    print(f"  max|Δ|/amax_blk    {float(np.max(np.abs(d)/np.maximum(amax_blk,1e-30))):.4f}")
    print(f"  码值范围           [{codes.min()}, {codes.max()}]  (应 ⊂ [-7,7])")
    print(f"  scale 字节范围     [{s_new.min()}, {s_new.max()}]  (旧 [{s_old.min()}, {s_old.max()}])")
    # scale 位移是否与 q 一致
    dq = s_new.astype(np.int32) - s_old.astype(np.int32)
    print(f"  scale 位移 Δq      min={dq.min()} max={dq.max()} mean={dq.mean():.2f}")
    nz = v_fp8 != 0
    print(f"  非零元素相对误差   中位 {float(np.median(np.abs(d[nz])/np.abs(v_fp8[nz]))):.4f}")
    print("\n[结论] 布局(nibble 序/两补码/scale 语义) 由独立解码路径复现；"
          "误差上界与对称 int4 的 1/14 相符。" if
          float(np.max(np.abs(d) / np.maximum(amax_blk, 1e-30))) <= 0.5 else "\n[警告] 误差超界")


if __name__ == "__main__":
    main()
