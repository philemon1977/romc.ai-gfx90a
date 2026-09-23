#!/usr/bin/env python3
"""engram 表 fp8(block32) → int4(block32) 就地重写。

为什么只动 2 个分片
------------------
DeepSeek-V4.1-Flash 的全部 engram 都在 `model-00047-of-00048`（layer 1）与
`model-00048-of-00048`（layer 14）两个文件里，各 5 个张量、~97 GiB：

    layers.{1,14}.engram.embed.weight   F8_E4M3  [384006168, 256]  91.55 GiB
    layers.{1,14}.engram.embed.scale    F8_E8M0  [384006168, 8]     2.86 GiB
    layers.{1,14}.engram.{k_weight,q_weight,wkv.weight}  BF16        0.35 GiB

其余 46 片每片 ≤7.4 GiB，与本脚本无关（新目录里对它们做 symlink 即可）。
engram 占整仓 189.4/494.6 = 38%，是唯一还能压缩的大块；4-bit 化后
23.6 → 12.2 GiB/rank，权重 61.8 → 50.3 GiB/rank，才能给 256K 上下文留出
KV 空间（这也是 v15–v21 试图用 host pinned+UVA offload 硬凑、却因宿主
只有 251 GiB 而反复 OOM 的那 189 GiB）。

量化为什么可以在"码字域"直接做
------------------------------
source 是 fp8 e4m3 + ue8m0(2 的幂) 的 32 元素块 scale：

    real = c_fp8 * 2^(s-127)        c_fp8 = e4m3 码解出的值

我们要的是同块最优对称 int4（15 级）scale S' = 2^m，S' ≥ amax_real/7：

    m = (s-127) + q,  q = 最小的整数使 7*2^q ≥ amax_codes
    code = round(c_fp8 / 2^q) ∈ [-7, 7]
    新 scale 字节 = s + q        (仍然 ue8m0，形状/类型不变)

q 用 frexp 精确求：amax = mant*2^e，mant∈[0.5,1) ⇒
    q = e-3 if mant ≤ 0.875 else e-2   (因为 7*2^(e-3) = 0.875*2^e)

即：**不需要反量化到 bf16 再重新做 amax 标定**，逐块整数/浮点一次即可，
且 scale 依旧保持 ue8m0（内核里 `(byte << 23)` bitcast 成 fp32 指数的技巧
可以原样复用）。

打包约定：U8 [rows, 128]，低 nibble 在偶数列（与 compressed-tensors 一致）,
int4 用两补码。
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import torch
from safetensors import safe_open

DT_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "F8_E8M0": 1, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4,
    "I64": 8, "U64": 8, "BOOL": 1,
}
ENGRAIN_SUFFIX = ".engram.embed.weight"


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def read_header(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n))


def build_spec(hdr):
    """name -> (out_dtype, out_shape, in_dtype, in_shape)，保持顺序。"""
    spec = []
    for name, info in hdr.items():
        if name == "__metadata__":
            continue
        dt, shape = info["dtype"], list(info["shape"])
        if name.endswith(ENGRAIN_SUFFIX):
            assert dt == "F8_E4M3", f"{name}: 期望 F8_E4M3，实测 {dt}"
            assert shape[1] % 2 == 0
            spec.append((name, "U8", [shape[0], shape[1] // 2], dt, shape))
        else:
            spec.append((name, dt, shape, dt, shape))
    return spec


def write_header(f, spec):
    off, out = 0, {}
    for name, dt, shape, _, _ in spec:
        nb = DT_BYTES[dt] * numel(shape)
        out[name] = (off, nb)
        off += nb
    hdr = {"__metadata__": {"format": "pt"}}
    total_off = 0
    for name, dt, shape, _, _ in spec:
        nb = DT_BYTES[dt] * numel(shape)
        hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [total_off, total_off + nb]}
        total_off += nb
    js = json.dumps(hdr, separators=(",", ":")).encode()
    js += b" " * ((-(8 + len(js))) % 8)
    f.write(struct.pack("<Q", len(js)))
    f.write(js)
    return 8 + len(js), out


def quant_chunk(w_fp8, s_u8, device, search=3):
    """w_fp8: [c, D] float8_e4m3fn(CPU) -> (packed U8 [c, D/2], new_scale U8 [c, D/32])

    scale 指数取块内 MSE 最小的那个：候选 q, q-1, ...
    纯 absmax 的 q 在有离群值的块里会把步长撑到 2 倍粗；往下试 1~2 档后
    少数被 clip 的离群值换来整块更细的步长，实测 SNR 有增益（见 --verify-rows）。
    """
    D = w_fp8.shape[1]
    BS = 32
    w = w_fp8.to(device=device, dtype=torch.float32)
    s = s_u8.to(device=device, dtype=torch.int32)
    blocks = w.view(w.shape[0], D // BS, BS)
    amax = blocks.abs().amax(dim=-1)                       # [c, D/BS]
    mant, exp = torch.frexp(amax)                          # amax = mant*2^exp
    base = torch.where(mant <= 0.875, exp - 3, exp - 2)    # 7*2^q >= amax
    best_q = best_err = None
    for k in range(max(1, search)):
        q = torch.where(amax > 0, base - k, torch.zeros_like(base))
        step = torch.exp2(q.to(torch.float32))[:, :, None]
        code_k = torch.round(blocks / step).clamp_(-7, 7)
        err = ((code_k * step - blocks) ** 2).sum(dim=-1)
        if best_err is None:
            best_q, best_err = q, err
        else:
            m = err < best_err
            best_q = torch.where(m, q, best_q)
            best_err = torch.where(m, err, best_err)
    q = best_q
    step = torch.exp2(q.to(torch.float32))[:, :, None]
    code = torch.round(blocks / step).clamp_(-7, 7).to(torch.int16)
    new_s = (s + q).clamp_(0, 254).to(torch.uint8)         # ue8m0 字节
    # 打包：偶数列在低 nibble
    c16 = code.to(torch.int32) & 0xF
    packed = (c16[:, :, 0::2] | (c16[:, :, 1::2] << 4)).to(torch.uint8)
    return packed.reshape(-1, D // 2).cpu(), new_s.cpu(), code, blocks


def run_shard(src, dst, args):
    hdr = read_header(src)
    spec = build_spec(hdr)
    big = [(i, s) for i, s in enumerate(spec) if s[0].endswith(ENGRAIN_SUFFIX)]
    assert len(big) == 1, f"{src}: engram.embed.weight 数量异常 {len(big)}"
    bi, (bname, _, bshape, _, inshape) = big[0]
    sname = bname[: -len("weight")] + "scale"
    si = [i for i, s in enumerate(spec) if s[0] == sname]
    assert len(si) == 1, f"{src}: 找不到 {sname}"
    si = si[0]

    rows = inshape[0]
    if args.limit_rows:
        rows = min(rows, args.limit_rows)

    t0 = time.time()
    dev = torch.device(args.device)
    written = 0
    stats = {"n": 0, "abs": 0.0, "maxrel": 0.0, "mx": 0.0}
    out_spec = list(spec)
    if args.limit_rows:
        out_spec[bi] = (bname, spec[bi][1], [rows, spec[bi][2][1]], spec[bi][3], spec[bi][4])
        out_spec[si] = (sname, spec[si][1], [rows, spec[si][2][1]], spec[si][3], spec[si][4])

    tmp = dst + ".tmp"
    with safe_open(src, framework="pt") as f, open(tmp, "wb") as out:
        data_start, offsets = write_header(out, out_spec)
        wsl = f.get_slice(bname)
        ssl = f.get_slice(sname)
        boff = data_start + offsets[bname][0]
        soff = data_start + offsets[sname][0]
        row_bytes_out = inshape[1] // 2
        srow_bytes = inshape[1] // 32
        for r0 in range(0, rows, args.chunk_rows):
            r1 = min(r0 + args.chunk_rows, rows)
            w = wsl[r0:r1]
            s = ssl[r0:r1].view(torch.uint8)
            packed, new_s, code, blocks = quant_chunk(w, s, dev, args.scale_search)
            out.seek(boff + r0 * row_bytes_out)
            out.write(packed.numpy().tobytes())
            out.seek(soff + r0 * srow_bytes)
            out.write(new_s.numpy().tobytes())
            if args.verify_rows and stats["n"] < args.verify_rows:
                # 反量化误差（取前若干行逐块精确算）
                v = blocks.double()
                dq = (code.double() * torch.exp2(
                    (new_s.to(dev).to(torch.int32) - 127).double())[:, :, None])
                err = (dq - v * torch.exp2(
                    (s.to(dev).to(torch.int32) - 127).double())[:, :, None]).abs()
                stats["n"] += err.numel()
                stats["abs"] += err.sum().item()
                stats["mx"] = max(stats["mx"], err.max().item())
            written += (r1 - r0) * (row_bytes_out + srow_bytes)
            if (r0 // args.chunk_rows) % 8 == 0:
                el = time.time() - t0
                print(f"  {r1}/{rows} rows  {written/2**30:.1f} GiB out  "
                      f"{el:.0f}s  {written/2**30/max(el,1e-9):.2f} GiB/s", flush=True)
        # 其余张量原样搬运
        for i, (name, dt, shape, idt, ishape) in enumerate(out_spec):
            if i == bi or i == si:
                continue
            t = f.get_tensor(name)
            if args.limit_rows:
                t = t[:rows]
            if t.dtype == torch.bfloat16:
                buf = t.view(torch.uint8).numpy().tobytes()
            elif t.dtype == torch.float8_e8m0fnu:
                buf = t.view(torch.uint8).numpy().tobytes()
            else:
                buf = t.contiguous().view(torch.uint8).numpy().tobytes()
            out.seek(data_start + offsets[name][0])
            out.write(buf)
            print(f"  copied {name} {dt} {tuple(t.shape)}", flush=True)
    os.replace(tmp, dst)
    el = time.time() - t0
    print(f"[done] {os.path.basename(dst)}  {el:.0f}s  {written/2**30:.1f} GiB", flush=True)
    if stats["n"]:
        print(f"[verify] int4 反量化 vs fp8 原始：mean|Δ|={stats['abs']/stats['n']:.5f} "
              f"max|Δ|={stats['mx']:.4f}  (样本 {stats['n']} 元素)", flush=True)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源 checkpoint 目录")
    ap.add_argument("--dst", required=True, help="输出目录（只写 47/48）")
    ap.add_argument("--shards", default="model-00047-of-00048.safetensors,"
                                        "model-00048-of-00048.safetensors")
    ap.add_argument("--chunk-rows", type=int, default=1_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-rows", type=int, default=0, help=">0 只处理前 N 行（自测用）")
    ap.add_argument("--verify-rows", type=int, default=0, help=">0 对首批块统计误差")
    ap.add_argument("--scale-search", type=int, default=3,
                    help="scale 指数候选档数（1=纯 absmax）")
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    tot0 = time.time()
    for sh in args.shards.split(","):
        sh = sh.strip()
        if not sh:
            continue
        print(f"=== {sh}", flush=True)
        run_shard(os.path.join(args.src, sh), os.path.join(args.dst, sh), args)
    print(f"[all done] {time.time()-tot0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
