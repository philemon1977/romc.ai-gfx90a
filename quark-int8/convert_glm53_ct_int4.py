#!/usr/bin/env python3
"""Convert DeepSeek-V4.1-Flash (FP4 experts + FP8 block-scaled linears) into a
**compressed-tensors W4A16 `pack-quantized`** checkpoint, so vLLM on gfx90a can
reach the ROCm W4A16 kernels (`TritonW4A16LinearKernel`) instead of the MXFP4
emulation path.

Why not Quark: vLLM's QuarkConfig implements only
``w8a8_fp8 / w8a8_int8 / ocp_mx / nvfp4 / w4a8_mxfp4_fp8`` — there is **no int4
weight-only scheme**, so a Quark int4 export is unloadable
(``NotImplementedError: No quark compatible scheme was found``).  Measured in
``quark-int8/INT4_CT_PLAN.md``.  The W4A16 kernels are consumed by
compressed-tensors / AWQ / GPTQ / moe_wna16 only.

Source layout (measured across all 48 shards):

  routed experts   ``layers.N.ffn.experts.E.{w1,w2,w3}``  I8 packed FP4
                   + ``.scale`` F8_E8M0, one row x 16 packed bytes
                   (= 1x32 fp4 elements), i.e. 0.5 B/elem.
                   Logical shape = (rows, packed_cols*2).
  fp8 linears      attention (wq_a/wq_b/wkv/wo_a/wo_b, indexer.wq_b),
                   GLM-5.3 源 = F8_E4M3 + ``.weight_scale_inv`` **F32** 乘数、块 128x128
                   （与本仓早期 DSV4.1 版的 fp4/F8_E8M0 32x32 完全不同，见 §4/§9 记录）
                   blocks, i.e. 1.0 B/elem.
  bf16             norms, routers (ffn.gate), embeddings, head, visual, biases.

Target layout per converted 2D weight (compressed-tensors conventions):
    <name>.weight_packed  int32 [N, K/8]      (8 x int4 per int32, low nibble first)
    <name>.weight_scale   float32 [N, K/32]   (symmetric int4, scale = amax/7)
    <name>.weight_shape   int64 [2] = (N, K)

Verified: ``pack_int4_g32`` output is byte-identical to
``compressed_tensors.compressors.PackedQuantizationCompressor.compress``.

Memory: one **module** at a time, on device, in row chunks, so peak device use
is bounded by ``--device-budget-gib`` regardless of tensor size.  Shards are
never held whole.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GROUP_SIZE = 32
# 源 fp8 量化块高（GLM-5.3 config: quantization_config.weight_block_size = [128,128]）
SRC_BLOCK = 128

# 逐组最优 scale 的搜索栅格（空 tuple = 关闭，走原来的 s=amax/7）。
# 动机与实测（见 int4_scale_optimality.py / 转换记录 §4.15 C）：
#   uint4b8 的栅格是**不对称**的 [-8s, +7s]，取 s=amax/7 虽不削顶，却把步长撑大；
#   对源权重逐组做 1 维 scale 搜索后，相对误差由 10.0% 降到 ~6.8%（1.47×）。
#   最优倍数实测稳定落在 0.90 附近，故栅格取 0.78..1.00、步长 0.02（覆盖两侧）。
_OPT_SCALE_FGRID: tuple[float, ...] = ()

# 共享专家是否留 bf16（见 classify 里的依据）
_SHARED_EXPERTS_BF16 = False
# GLM: 默认共享专家 bf16（量少、质量敏感）
_SHARED_EXPERTS_INT4 = os.environ.get("GLM_SHARED_INT4", "0") == "1"

# 注意力三个**非合并**投影是否留 bf16（wq_a/wkv 是合并模块 fused_wqa_wkv，不动）
_ATTN_BF16 = False

# 合并模块 fused_wqa_wkv 的两个源张量（wq_a/wkv）是否也留 bf16。
# 依据（§4.19 F，读 load_weights 实测代码）：合并模块的 checkpoint 张量**可以是分开的**
# wq_a.weight/wkv.weight —— 加载器按 stacked_params_mapping + shard_id 写入合并参数的对应段。
# §4.10 D 那次 KeyError 的真因是 ignore 写了 checkpoint 名（没命中模块名 ⇒ 模块仍量化 ⇒
# params 里只有 .weight_packed）。故这里 ignore 必须写**模块名** *attn.fused_wqa_wkv。
_ATTN_MERGED_BF16 = False


# ---------------------------------------------------------------- precision policy
def st_tensor_names(path: str) -> list[str]:
    """Tensor names in a safetensors file, header-only (no tensor data)."""
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            h = json.loads(f.read(n))
        return [k for k in h if k != "__metadata__"]
    except Exception:
        return []


def index_from_output(out_dir: str) -> dict[str, str]:
    """Build the weight map from every shard present in ``out_dir``.

    Needed because a run may only (re)write a subset of shards: the index must
    still describe the whole checkpoint, otherwise it silently shrinks.
    """
    m: dict[str, str] = {}
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".safetensors"):
            continue
        for k in st_tensor_names(os.path.join(out_dir, fn)):
            m[k] = fn
    return m


def classify(module: str) -> str:
    """GLM-5.3（glm_moe_dsa）精度策略。module 为去掉 .weight/.scale_inv 的模块名。

    依据（本机实测体量，见 quark-int8/glm53_sizing.py）：
      路由专家 = 753.3G 参数里的 97.5% ⇒ 只有 int4（0.5+0.0625 B/参数）能装进 496 GiB 可用 HBM；
      全 int8 = 701.6 GiB ✗ 装不下；Quark 的 int4 是 OCP-MX(需 fp4 硬件) ✗ gfx90a 无。
      ⇒ 走我们已验证的 compressed-tensors W4A16（Triton W4A16 + WNA16 MoE + 我们的 GEMV 补丁）。
    源权重是 **fp8 块量化**（无 fp4 的 nibble 序问题），反量化路径只有 dequant_fp8_block 一条。
    """
    # 1) 路由专家 → int4（主体）
    if re.search(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj$", module):
        return "fp8_block"
    # 1b) ★ dense 前若干层的 MLP（GLM config: first_k_dense_replace=3）也必须是 nn.Linear
    #     ⇒ vLLM 会按 CT 配置（targets=Linear）把它当量化层建；漏掉它就会 KeyError:
    #     'layers.0.mlp.down_proj.weight'（2026-09-20 首次起服实测）。
    if re.search(r"\.mlp\.(gate|up|down)_proj$", module):
        return "fp8_block"
    # 2) 共享专家：默认 bf16（仅 228 个张量，质量敏感；GLM_SHARED_INT4=1 可改 int4）
    if re.search(r"\.mlp\.shared_experts\.(gate|up|down)_proj$", module):
        return "fp8_block" if _SHARED_EXPERTS_INT4 else "fp8_to_bf16"
    # 3) 注意力投影 → int4（吃 W4A16 线性内核；--attn-bf16 升 bf16）
    if re.search(r"\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$", module):
        return "fp8_to_bf16" if _ATTN_BF16 else "fp8_block"
    # 4) DSA 稀疏 indexer：
    #    ★ wk 在 vLLM 里与 weights_proj **融合成 indexer.wk_weights_proj** 一个模块，
    #      融合模块只能有一种 scheme ⇒ 不能只量化一半（2026-09-20 实测：
    #      KeyError: 'layers.0.self_attn.indexer.wk_weights_proj.weight_packed'）。
    #      处置与 DSV4.1 已验证配方一致：wk 与 weights_proj 都保持未量化，
    #      但源里 wk 是 fp8 ⇒ 必须反量化成 bf16 再写。
    if re.search(r"\.self_attn\.indexer\.wk$", module):
        return "fp8_to_bf16"
    if re.search(r"\.self_attn\.indexer\.wq_b$", module):
        return "fp8_block"
    # 5) 其余（norm / 路由器 gate / weights_proj / 嵌入 / lm_head / 打分偏置）保持原精度
    return "keep"


# ---------------------------------------------------------------- dequantizers
def e8m0_to_f32(scale: torch.Tensor) -> torch.Tensor:
    """F8_E8M0 (value = 2**(byte-127)) -> float32."""
    u = scale.view(torch.uint8).to(torch.int32) - 127
    return torch.exp2(u.float())


_FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]  # E2M1 magnitudes


def dequant_fp4_expert(packed: torch.Tensor, scale: torch.Tensor,
                       dtype=torch.bfloat16) -> torch.Tensor:
    """I8 packed MXFP4 + 1x32 E8M0 scale -> dense (rows, packed_cols*2).

    Two FP4 nibbles per byte; ``_is_mxfp4_source_pattern`` in Quark's own
    file-to-file path documents this DSV4 convention (scale_width ==
    packed_width/16, i.e. 32 FP4 elements per e8m0 scale).
    """
    assert packed.dtype == torch.int8, packed.dtype
    assert packed.dim() == 2 and scale.dim() == 2, (packed.shape, scale.shape)
    assert packed.shape[0] == scale.shape[0], (packed.shape, scale.shape)
    assert packed.shape[1] == scale.shape[1] * 16, (packed.shape, scale.shape)
    b = packed.contiguous().view(torch.uint8)
    # ★★ nibble 序 = 低 nibble 为偶数元素（2026-09-20 修正）
    #   权威依据（两处独立）:
    #     1) 源模型目录自带的官方 inference/convert.py::cast_e2m1fn_to_e4m3fn
    #        low = x & 0x0F ; high = (x >> 4) & 0x0F ; stack([FP4_TABLE[low], FP4_TABLE[high]])
    #     2) 运行时解包 vLLM .../compressed_tensors_moe_w4a16_flydsl.py::
    #        _unpack_gptq_int32_to_signed_int4 —— shifts = arange(8)*4, nibbles-8（低 nibble 先）
    #   原实现写成 hi=偶数(即相邻元素对互换)，使全部 CT-int4 仓的 47,232 个专家权重
    #   成为源行的「相邻对互换」版本：模型仍流畅但质量退化（复读吸引子），
    #   而官方编码臂（同一批源文件、vLLM 自带 loader）健康。详见
    #   /home/qiba/ai/docs/DeepSeek-V4.1-Flash-CT-INT4-W4A16-转换记录-2026-09-18.md §4.34
    lo = b & 0x0F                              # element 2j
    hi = (b >> 4) & 0x0F                       # element 2j+1
    q = torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)
    lut = torch.tensor(_FP4_LUT, dtype=torch.float32, device=packed.device)
    val = lut[(q & 0x07).long()]
    val = torch.where((q & 0x08) != 0, -val, val)
    s = e8m0_to_f32(scale).repeat_interleave(32, dim=1)
    assert s.shape == val.shape, (s.shape, val.shape)
    return (val * s).to(dtype)


def dequant_fp8_block(w: torch.Tensor, scale: torch.Tensor,
                      dtype=torch.bfloat16) -> torch.Tensor:
    """F8_E4M3 + 块 scale -> dense (N, K)。

    ★ 两种源并存（2026-09-20 实测）：
      - DSV4.1: scale 是 F8_E8M0 指数字节，乘数 = 2^(byte-127)
      - GLM-5.3: scale 是 **F32**，本身就是乘数（weight_scale_inv，128x128 块）
    混用会让形状/数量级同时错（F32 张量按 uint8 view 会多出 4 倍列）。
    """
    return _dequant_fp8_block(w, scale, dtype, SRC_BLOCK)


def _dequant_fp8_block(w: torch.Tensor, scale: torch.Tensor,
                       dtype=torch.bfloat16, block: int = 128) -> torch.Tensor:
    """固定块高展开（★ 处理"最后一块不满"）。"""
    assert w.dim() == 2 and scale.dim() == 2, (w.shape, scale.shape)
    n, k = w.shape
    nb = -(-n // block)      # 期望的 scale 行数 = ceil(n/block)
    assert scale.shape[0] == nb, (w.shape, scale.shape, "块高应为 %d" % block)
    assert k % block == 0, (w.shape, "列块不整除")
    if scale.dtype in (torch.uint8, torch.float8_e8m0fnu):
        s = e8m0_to_f32(scale)
    else:                                    # GLM: F32 块 scale，已是乘数
        s = scale.to(torch.float32)
    s = s.repeat_interleave(block, dim=0)[:n].repeat_interleave(block, dim=1)[:, :k]
    return (w.to(torch.float32) * s).to(dtype)


# ---------------------------------------------------------------- quantizer / packer
def pack_int4_rows(w: torch.Tensor, group_size: int = GROUP_SIZE,
                   scale_dtype: torch.dtype = torch.bfloat16
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric RTN int4 (qmax=7), per-group along K, for a row chunk of ``w``.

    Returns ``(packed int32 [n, K/8], scale [n, K/group_size])``.

    The scale is emitted in ``scale_dtype`` (bf16) because vLLM builds its
    ``weight_scale`` parameter with ``params_dtype``, which is bf16 for this
    model.  Shipping fp32 forces a dtype conversion for *every* scale tensor at
    load time (47,585 tensors, ~17.5e9 elements) and dominated startup time.
    bf16 costs 0.389% max relative error on a value whose int4 quantisation
    error is already ~20%, so it is effectively free.
    """
    n, k = w.shape
    assert k % group_size == 0, f"K={k} not divisible by group_size={group_size}"
    wg = w.to(torch.float32).reshape(n, k // group_size, group_size)
    scale = (wg.abs().amax(dim=-1) / 7.0).clamp_min(1e-8)
    if _OPT_SCALE_FGRID:
        # 逐组 1 维搜索：在 s0=amax/7 的倍数上取组内平方误差最小者。
        # 允许轻微削顶以换取更小的步长——这是 10.0%→6.8% 的来源（实测，非估计）。
        s0 = scale.unsqueeze(-1)
        best_err = None
        best_f = None
        for f in _OPT_SCALE_FGRID:
            s = s0 * f
            err = (torch.round(wg / s).clamp_(-8, 7) * s - wg).pow(2).sum(dim=-1)
            if best_err is None:
                best_err, best_f = err, torch.full_like(err, float(f))
            else:
                m = err < best_err
                # ★ torch.where(cond, A, B) 是「cond 为真取 A」。本轮更优时要取**本轮**的 f；
                #   第一版写反成 (m, best_f, 新f)，语义变成「更优则保留旧 f、更差则采用新 f」
                #   ⇒ 选出的 f 与最优无关，实测 --optimal-scale 反而变差（10.08%→10.15%）。
                best_err = torch.where(m, err, best_err)
                best_f = torch.where(m, torch.full_like(err, float(f)), best_f)
        scale = (scale * best_f).clamp_min(1e-8)
        del best_err, best_f, s0
    q = torch.round(wg / scale.unsqueeze(-1)).clamp_(-8, 7).to(torch.int32).reshape(n, k) + 8
    q = q.reshape(n, k // 8, 8)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=q.device)
    return (q << shifts).sum(dim=-1, dtype=torch.int32).contiguous(), \
        scale.to(scale_dtype).contiguous()


_ST_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
             "F8_E8M0": 1, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4,
             "I64": 8, "U64": 8, "BOOL": 1}


def get_passthrough(f, name: str, max_gib: float) -> torch.Tensor:
    """Read a tensor we keep verbatim, in row chunks when it is huge.

    The engram tables are 94 GiB each; pulling one whole plus holding the output
    copy peaked host RAM at 183/251 GiB, uncomfortably close to the OOM killer.
    Chunked reads return one tensor while bounding peak.
    """
    sl = f.get_slice(name)
    shape = sl.get_shape()
    if not shape or max_gib <= 0:
        return f.get_tensor(name)
    row_elems = 1
    for d in shape[1:]:
        row_elems *= d
    row_bytes = row_elems * _ST_BYTES.get(str(sl.get_dtype()), 1)
    rows_per_chunk = max(1, int(max_gib * (1 << 30)) // max(row_bytes, 1))
    if rows_per_chunk >= shape[0]:
        return f.get_tensor(name)
    first = sl[0:min(rows_per_chunk, shape[0])]
    out = torch.empty(shape, dtype=first.dtype)
    out[0:first.shape[0]] = first
    del first
    for i in range(rows_per_chunk, shape[0], rows_per_chunk):
        j = min(i + rows_per_chunk, shape[0])
        out[i:j] = sl[i:j]
    return out


_DRM_BY_PCI: dict[str, int] | None = None
def _drm_index_by_pci() -> dict[str, int]:
    """Map PCI BDF -> the /sys/class/drm/cardN index, built once."""
    global _DRM_BY_PCI
    if _DRM_BY_PCI is None:
        m: dict[str, int] = {}
        try:
            for name in os.listdir("/sys/class/drm"):
                p = f"/sys/class/drm/{name}/device"
                if not name.startswith("card") or "-" in name or not os.path.exists(p):
                    continue
                bdf = os.path.basename(os.path.realpath(p))
                m[bdf] = int(name[4:])
        except OSError:
            pass
        _DRM_BY_PCI = m
    return _DRM_BY_PCI


def _visible_bdfs() -> list[str]:
    """PCI BDFs of the GPUs this process may use, in torch index order."""
    vis = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("ROCR_VISIBLE_DEVICES")
    all_bdfs = sorted(_drm_index_by_pci().keys())
    if not vis:
        return all_bdfs
    out = []
    for tok in vis.split(","):
        tok = tok.strip()
        if tok.isdigit():
            i = int(tok)
            if 0 <= i < len(all_bdfs):
                out.append(all_bdfs[i])
        else:
            out.append(tok)
    return out


def _free_gib_for_bdf(bdf: str) -> float | None:
    card = _drm_index_by_pci().get(bdf)
    if card is None:
        return None
    try:
        with open(f"/sys/class/drm/card{card}/device/mem_info_vram_used") as fu, \
             open(f"/sys/class/drm/card{card}/device/mem_info_vram_total") as ft:
            return (int(ft.read()) - int(fu.read())) / (1 << 30)
    except OSError:
        return None


def visible_free_gib() -> list[float | None]:
    """Free VRAM per visible GPU in torch index order; None where unknown."""
    return [_free_gib_for_bdf(b) for b in _visible_bdfs()]


def device_free_gib(dev: torch.device) -> float:
    """Free device memory in GiB, via sysfs when possible.

    Reading sysfs avoids initialising the HIP context just to ask a question,
    which matters when a co-tenant is using the GPU.
    """
    if dev.type != "cuda":
        return float("inf")
    idx = dev.index or 0
    bdfs = _visible_bdfs()
    if 0 <= idx < len(bdfs):
        v = _free_gib_for_bdf(bdfs[idx])
        if v is not None:
            return v
    free, _total = torch.cuda.mem_get_info(dev)
    return free / (1 << 30)


def _wait_for_memory(dev: torch.device, needed_gib: float, what: str,
                     poll_s: float, max_s: float, enable: bool) -> bool:
    """Block until the device has ``needed_gib`` free, or give up.

    A co-tenant's footprint fluctuates (KV-cache growth, request bursts), so a
    hard failure on one tight moment is wrong: pause and resume instead.
    """
    if dev.type != "cuda":
        return True
    t0 = time.time()
    waited = 0.0
    while True:
        free = device_free_gib(dev)
        if free >= needed_gib:
            if waited:
                print(f"[convert]   resumed: {free:.1f} GiB free after waiting {waited:.0f}s",
                      flush=True)
            return True
        if not enable or (time.time() - t0) > max_s:
            return False
        if waited == 0.0 or (time.time() - t0) - waited >= 30:
            print(f"[convert]   waiting for GPU: {free:.1f} GiB free < {needed_gib} GiB "
                  f"needed{what} (co-tenant); will retry", flush=True)
        time.sleep(poll_s)
        waited = time.time() - t0
        if dev.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def convert_module(base: str, w: torch.Tensor, s: torch.Tensor, kind: str,
                   dev: torch.device, dt: torch.dtype,
                   row_chunk: int, min_free_gib: float = 0.0,
                   wait_max_s: float = 0.0, wait_poll_s: float = 15.0) -> dict[str, torch.Tensor]:
    """Dequantize one module in row chunks, requantize to int4.

    Chunk size adapts to live free device memory each iteration, halves on OOM,
    and waits for a memory window when a co-tenant is holding the device.
    """
    n, k_logical = w.shape[0], (w.shape[1] * 2 if kind == "fp4_expert" else w.shape[1])
    assert k_logical % GROUP_SIZE == 0, f"{base}: K={k_logical} not divisible by {GROUP_SIZE}"
    # device bytes per row: packed src (I8) + scale src (F8) + fp32 dense + int4 q
    bytes_per_row = w.shape[1] + s.shape[1] + k_logical * 4 + k_logical * 4
    packed_rows, scale_rows = [], []
    r0 = 0
    # ★ 行分块必须与源 scale 的**行块**对齐（GLM-5.3: 128 行/块），否则
    #   dequant_fp8_block 里 s[r0:r1] 与 w[r0:r1] 的块对应关系断裂
    #   （实测报错：w(576,6144) vs s(5,48) ⇒ 576/5 不是整数块）。
    src_blk = SRC_BLOCK if (s.dim() == 2 and s.shape[0]) else 1
    row_chunk = max(src_blk, (row_chunk // src_blk) * src_blk)
    chunk = row_chunk
    while r0 < n:
        if dev.type == "cuda":
            if not _wait_for_memory(dev, min_free_gib, f" for {base}",
                                    wait_poll_s, wait_max_s, wait_max_s > 0):
                free = device_free_gib(dev)
                raise torch.OutOfMemoryError(
                    f"device has only {free:.1f} GiB free (< {min_free_gib} GiB guard) "
                    f"at {base} row {r0}/{n} and no window opened within {wait_max_s:.0f}s")
            free = device_free_gib(dev)
            chunk = max(1, min(row_chunk, int(free * (1 << 30) * 0.25) // max(bytes_per_row, 1)))
        r1 = min(r0 + chunk, n)
        try:
            wd = w[r0:r1].to(dev)
            # ★ scale 要按**块下标**切片（r0 已是块高对齐的）：行 [r0,r1) 对应 scale 行
            #   [r0/BLOCK, ceil(r1/BLOCK))。原实现用 s[r0:r1] 会取错（GLM 的 scale 行数
            #   远少于权重行数：576 行权重只有 5 行 scale）。
            if kind == "fp4_expert":
                sd = s[r0:r1].to(dev)
            else:
                _b = max(1, SRC_BLOCK)
                sd = s[r0 // _b: -(-r1 // _b)].to(dev)
            dense = (dequant_fp4_expert(wd, sd, dt) if kind == "fp4_expert"
                     else dequant_fp8_block(wd, sd, dt))
            pk, sc = pack_int4_rows(dense)
        except torch.OutOfMemoryError:
            if chunk <= 1:
                raise
            chunk = max(1, chunk // 4)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[convert]   OOM on {base} row {r0}; retrying with chunk={chunk}",
                  flush=True)
            continue
        packed_rows.append(pk.cpu())
        scale_rows.append(sc.cpu())
        del wd, sd, dense, pk, sc
        r0 = r1
    out = {
        f"{base}.weight_packed": torch.cat(packed_rows, dim=0),
        f"{base}.weight_scale": torch.cat(scale_rows, dim=0),
        f"{base}.weight_shape": torch.tensor([n, k_logical], dtype=torch.int64),
    }
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--device", default="cuda", help="cuda | cuda:N | cpu")
    ap.add_argument("--device-budget-gib", type=float, default=12.0,
                    help="approx device memory budget for dequant chunking")
    ap.add_argument("--min-free-gib", type=float, default=2.0,
                    help="refuse to use the GPU when less than this is free "
                         "(co-tenant guard); with --allow-cpu it falls back to CPU")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="fall back to CPU when the GPU has no room (slower, but "
                         "does not disturb other GPU tenants)")
    ap.add_argument("--wait-max-s", type=float, default=1800.0,
                    help="when a co-tenant holds the GPU, wait up to this long for a "
                         "memory window before giving up (0 disables waiting)")
    ap.add_argument("--wait-poll-s", type=float, default=15.0,
                    help="poll interval while waiting for GPU memory")
    ap.add_argument("--no-wait-for-gpu", action="store_true",
                    help="fail fast instead of waiting for a co-tenant to free memory")
    ap.add_argument("--max-keep-gib", type=float, default=16.0,
                    help="read pass-through tensors larger than this in row chunks "
                         "(bounds host RAM on the 94 GiB engram shards; 0 disables)")
    ap.add_argument("--shards", default="", help="comma-separated shard numbers; empty = all")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--attn-merged-bf16", action="store_true",
                    help="合并模块 fused_wqa_wkv 的 wq_a/wkv 也留 bf16（checkpoint 保持分开命名，"
                         "ignore 写模块名 *attn.fused_wqa_wkv）。实测它占注意力参数 7.2% 却贡献 4.62% 输出偏差")
    ap.add_argument("--attn-bf16", action="store_true",
                    help="注意力 wq_b/wo_a/wo_b（非合并投影）留 bf16；实测注意力输出 int4 损伤 10.687%%，"
                         "而它占该层输出的 57%%（见 quant_damage_attn.py）")
    ap.add_argument("--shared-experts-bf16", action="store_true",
                    help="共享专家留 bf16 不量化（依据：它贡献是路由专家的 4.9 倍，"
                         "却只占 1.47%% 的数值量、代价 +0.25 GiB/rank）")
    ap.add_argument("--optimal-scale", action="store_true",
                    help="逐组搜索最优 int4 scale（实测相对误差 10.0%% -> ~6.8%%；"
                         "见 _OPT_SCALE_FGRID 注释）")
    ap.add_argument("--opt-scale-lo", type=float, default=0.72)
    ap.add_argument("--opt-scale-hi", type=float, default=1.00)
    ap.add_argument("--opt-scale-step", type=float, default=0.05)  # 0.02 更准但慢 2.2×；实测 0.05 仅变差 +1.887%（§4.20）
    args = ap.parse_args()

    if args.attn_merged_bf16:
        global _ATTN_MERGED_BF16
        _ATTN_MERGED_BF16 = True
        print("[convert] 合并模块 wq_a/wkv: bf16（分开命名 + ignore 用模块名 *attn.fused_wqa_wkv）",
              flush=True)

    if args.attn_bf16:
        global _ATTN_BF16
        _ATTN_BF16 = True
        print("[convert] 注意力 wq_b/wo_a/wo_b: bf16"
              + ("；wq_a/wkv 亦 bf16（合并模块，见 --attn-merged-bf16）" if _ATTN_MERGED_BF16
                 else "；wq_a/wkv 保持 int4"), flush=True)

    if args.shared_experts_bf16:
        global _SHARED_EXPERTS_BF16
        _SHARED_EXPERTS_BF16 = True
        print("[convert] 共享专家: bf16（不量化）", flush=True)

    if args.optimal_scale:
        global _OPT_SCALE_FGRID
        grid, f = [], args.opt_scale_lo
        while f <= args.opt_scale_hi + 1e-9:
            grid.append(round(f, 4))
            f += args.opt_scale_step
        _OPT_SCALE_FGRID = tuple(grid)
        print(f"[convert] 逐组最优 scale: 开（{len(grid)} 点，{grid[0]:.2f}..{grid[-1]:.2f}）"
              f" —— 允许轻微削顶换更小步长", flush=True)

    if args.group_size != GROUP_SIZE:
        raise SystemExit(f"this converter implements group_size={GROUP_SIZE} only "
                         f"(got {args.group_size})")
    dev = torch.device(args.device)
    budget = int(args.device_budget_gib * (1 << 30))
    max_keep_gib = args.max_keep_gib
    # Co-tenant guard: another session may hold most of (or all) the devices.
    if dev.type == "cuda" and dev.index is None:
        frees = visible_free_gib()
        known = [(i, v) for i, v in enumerate(frees) if v is not None]
        if known:
            print(f"[convert] free VRAM per visible GPU (GiB): "
                  f"{', '.join(f'{i}:{v:.1f}' for i, v in known)}")
            best, best_free = max(known, key=lambda kv: kv[1])
            if best_free < args.min_free_gib and not args.no_wait_for_gpu:
                # wait for a window on the least-loaded GPU
                cand = torch.device(f"cuda:{best}")
                if _wait_for_memory(cand, args.min_free_gib,
                                    f" to start on gpu{best}", args.wait_poll_s,
                                    args.wait_max_s, True):
                    best_free = device_free_gib(cand)
            if best_free >= args.min_free_gib:
                print(f"[convert] selecting least-loaded GPU {best} ({best_free:.1f} GiB free)")
                dev = torch.device(f"cuda:{best}")
            else:
                msg = (f"no visible GPU has {args.min_free_gib} GiB free "
                       f"(best is gpu{best} with {best_free:.1f} GiB)")
                if args.allow_cpu:
                    print(f"[convert] WARNING: {msg} -> falling back to CPU "
                          f"(slower; leaves GPU tenants untouched)")
                    dev = torch.device("cpu")
                else:
                    raise SystemExit(
                        f"[convert] {msg}. Another session is probably using the GPUs. "
                        f"Re-run with --allow-cpu, or wait and re-run with "
                        f"--skip-existing to resume.")
    os.makedirs(args.out, exist_ok=True)

    idx = json.load(open(os.path.join(args.model, "model.safetensors.index.json")))
    wmap: dict[str, str] = idx["weight_map"]
    shard_num = lambda s: int(re.search(r"-(\d+)-of-", s).group(1))
    all_shards = sorted(set(wmap.values()), key=shard_num)

    by_shard: dict[str, list[str]] = {s: [] for s in all_shards}
    for nm, s in wmap.items():
        by_shard[s].append(nm)

    if args.shards:
        want = {int(x) for x in args.shards.split(",")}
        all_shards = [s for s in all_shards if shard_num(s) in want]

    def modules_of(shard: str) -> dict[str, dict[str, str]]:
        """canonical module name -> {'weight'|'scale': full tensor name}"""
        g: dict[str, dict[str, str]] = {}
        for nm in by_shard[shard]:
            # ★ GLM-5.3 的块 scale 叫 ".weight_scale_inv"（不是 DSV4.1 fp4 的 ".scale"）。
            #   原实现只认 ".weight"/".scale" ⇒ scale 被当成独立模块的自有张量 ⇒
            #   量化分支因 "scale" not in leaves 全部退回 keep
            #   （试点实测：plan 说 500 个 fp8_block，实际 0 个被转换、产物 0 个 weight_packed）。
            if nm.endswith(".weight_scale_inv"):
                g.setdefault(nm[: -len(".weight_scale_inv")], {})["scale"] = nm
            elif nm.endswith(".weight"):
                g.setdefault(nm[: -len(".weight")], {})["weight"] = nm
            elif nm.endswith(".scale"):
                g.setdefault(nm[: -len(".scale")], {})["scale"] = nm
            else:
                g.setdefault(nm, {})["self"] = nm
        return g

    # ---- inventory -------------------------------------------------------
    tot: dict[str, int] = {}
    expected = 0
    for shard in all_shards:
        for module, leaves in modules_of(shard).items():
            kind = classify(module)
            if kind == "keep" or "weight" not in leaves or len(leaves.get("weight", "")) == 0:
                tot["keep"] = tot.get("keep", 0) + 1
                continue
            tot[kind] = tot.get(kind, 0) + 1
            expected += 1

    print(f"[convert] source : {args.model}")
    print(f"[convert] out    : {args.out}")
    print(f"[convert] shards : {len(all_shards)}  device={dev}  budget={args.device_budget_gib} GiB")
    print(f"[convert] plan   : {tot}   (to convert: {expected})")
    # ★ 文案与 GLM 实际策略一致（原来整段是 DSV4.1 的，会误导日志读者）
    print(f"[convert] policy : fp8_block=FP8(块 {SRC_BLOCK}x{SRC_BLOCK}, F32 乘数 scale)->int4(g{GROUP_SIZE})  "
          f"fp8_to_bf16=反量化保 bf16  keep=byte-identical")
    print("[convert] 注意 : --attn-merged-bf16 是 DSV4.1 专用开关（GLM 无 fused_wqa_wkv 模块），对本模型无效")
    if args.dry_run:
        return
    # Guard against the silent 'classification is broken' failure seen in the
    # first pilot (zero modules matched).  Only meaningful for a full sweep:
    # a targeted subset (e.g. re-converting the engram-only shards, which are
    # now pass-through) legitimately has nothing to convert.
    if expected == 0 and not args.shards:
        raise SystemExit("[convert] ERROR: nothing to convert — classification is broken")
    if expected == 0:
        print("[convert] NOTE: this shard subset has nothing to convert "
              "(all pass-through); refreshing outputs only")

    index: dict[str, str] = {}
    stats = {"fp4_expert": 0, "fp8_block": 0, "keep": 0, "fp8_to_bf16": 0}
    t0 = time.time()
    for si, shard in enumerate(all_shards, 1):
        out_path = os.path.join(args.out, shard)
        if args.skip_existing and os.path.exists(out_path):
            for k in st_tensor_names(out_path):
                index[k] = shard
            print(f"[convert] {si}/{len(all_shards)} {shard}: skipped (exists)", flush=True)
            continue

        modules = modules_of(shard)
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(os.path.join(args.model, shard), framework="pt") as f:
            for module, leaves in modules.items():
                kind = classify(module)
                if kind == "keep":
                    for nm in leaves.values():
                        out_tensors[nm] = get_passthrough(f, nm, max_keep_gib)
                    stats["keep"] += 1
                    continue
                if kind == "fp8_to_bf16":
                    if "weight" not in leaves or "scale" not in leaves:
                        for nm in leaves.values():
                            out_tensors[nm] = get_passthrough(f, nm, max_keep_gib)
                        stats["keep"] += 1
                        continue
                    dense = dequant_fp8_block(
                        f.get_tensor(leaves["weight"]), f.get_tensor(leaves["scale"]),
                        torch.bfloat16)
                    out_tensors[leaves["weight"]] = dense
                    stats["fp8_to_bf16"] += 1
                    continue
                if "weight" not in leaves or "scale" not in leaves:
                    for nm in leaves.values():
                        out_tensors[nm] = get_passthrough(f, nm, max_keep_gib)
                    stats["keep"] += 1
                    continue

                w = f.get_tensor(leaves["weight"])
                s = f.get_tensor(leaves["scale"])
                if w.dim() != 2:
                    out_tensors[leaves["weight"]] = w
                    out_tensors[leaves["scale"]] = s
                    stats["keep"] += 1
                    continue

                n, kb = w.shape
                k_logical = kb * 2 if kind == "fp4_expert" else kb
                # rows per chunk so that (w + scale + dense + q) stays in budget
                bytes_per_row = k_logical * 4 * 3 + kb * 6
                row_chunk = max(1, min(n, budget // max(bytes_per_row, 1)))
                try:
                    out_tensors.update(convert_module(module, w, s, kind, dev,
                                                      torch.bfloat16, row_chunk,
                                                      min_free_gib=args.min_free_gib,
                                                      wait_max_s=0.0 if args.no_wait_for_gpu else args.wait_max_s,
                                                      wait_poll_s=args.wait_poll_s))
                except Exception as _e:  # noqa: BLE001
                    # 不留静默回退：退回 keep 会让 config 仍声称 int4 ⇒ 装载时缺 weight_packed。
                    # 这里补上模块名与形状，便于定位（宁可响亮失败，也不要产出不自洽的仓）。
                    raise RuntimeError(
                        f"{module}: 量化失败 w={tuple(w.shape)} s={tuple(s.shape)} "
                        f"s.dtype={s.dtype} kind={kind}") from _e
                stats[kind] += 1
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

        save_file(out_tensors, out_path, metadata={"format": "pt"})
        for k in out_tensors:
            index[k] = shard
        print(f"[convert] {si}/{len(all_shards)} {shard}: {len(out_tensors)} tensors  "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)

    # ---- config.json -----------------------------------------------------
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    ignore = [
        "lm_head", "head", "embed",
        "*norm*",                       # 所有 RMSNorm/LayerNorm（含 kv_a/q_a_layernorm、indexer.k_norm）
        "*mlp.gate",                    # ★ 只匹配路由器模块（...mlp.gate）。
                                        #   写成 "*mlp.gate*" 会误命中 dense 层的 mlp.gate_proj
                                        #   ⇒ 被当未量化构建，而仓里是 fp8 ⇒ 静默 dtype 错（2026-09-20 实测）
        "*self_attn.indexer.weights_proj*",
        "*self_attn.indexer.wk*",       # ★ 同时覆盖 wk 本体与融合名 wk_weights_proj
        "*shared_expert_gate*",
        "*eh_proj",                     # ★ MTP(layer 78) 的 eh_proj：源就是 bf16，必须按未量化构建
                                        #   （dense MLP 那次起服失败的同型问题）
    ]
    if not _SHARED_EXPERTS_INT4:
        ignore += ["*mlp.shared_experts.*"]
    if _ATTN_BF16:
        ignore += ["*self_attn.q_a_proj", "*self_attn.q_b_proj",
                   "*self_attn.kv_a_proj_with_mqa", "*self_attn.kv_b_proj",
                   "*self_attn.o_proj"]
    if _ATTN_MERGED_BF16:
        # ★ 必须写**模块名**（合并后的 fused_wqa_wkv），写 checkpoint 名不会命中 ⇒ 重演 §4.10 D
        ignore += ["*attn.fused_wqa_wkv"]
    if _ATTN_BF16:
        ignore += ["*attn.wq_b", "*attn.wo_a", "*attn.wo_b", "*attn.indexer.wq_b"]
    if _SHARED_EXPERTS_BF16:
        # 通配同时覆盖 shared_experts.* 与 shared_expert_gate（后者本就该忽略）
        ignore += ["*shared_expert*"]
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "input_activations": None,
                "output_activations": None,
                "weights": {
                    "num_bits": 4, "type": "int", "symmetric": True,
                    "strategy": "group", "group_size": GROUP_SIZE,
                    "dynamic": False, "actorder": None,
                },
            }
        },
        "ignore": ignore,
        "quantization_status": "compressed",
    }
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    for fn in os.listdir(args.model):
        if fn.endswith(".safetensors") or fn.startswith("model.safetensors.index"):
            continue
        src, dst = os.path.join(args.model, fn), os.path.join(args.out, fn)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Merge every shard on disk: a subset run must not shrink the index.
    on_disk = index_from_output(args.out)
    if len(on_disk) != len(index):
        print(f"[convert] index merge: run covered {len(index)} tensors, "
              f"on disk {len(on_disk)}; writing the union")
    index = on_disk
    json.dump({"metadata": {"total_size": 0}, "weight_map": index},
              open(os.path.join(args.out, "model.safetensors.index.json"), "w"), indent=1)

    # Count converted modules from the *final index*, not from this run's
    # counters: with --skip-existing, shards converted by an earlier run are
    # already in the index but were never re-processed here.  Finalisation above
    # must therefore happen before this sanity check, never after it.
    #
    # ★ 同一陷阱的第二处：回读"已存在分片"来统计 converted 数时，`fp8_to_bf16` 模块
    #   既没有 weight_packed 也没有 weight_shape，只留一个 bf16 `.weight` ⇒ 旧逻辑认不出
    #   ⇒ 全 bf16 化跑完后 self-check 报 `ERROR: 47235 vs 47587`（差额恰等于 bf16 模块数），
    #   但产物逐个模块核对是完整的（audit_checkpoint_complete.py：缺失=0、冗余=0）。
    #   即：**计数是代理指标，别拿它当正确性判据**；这里补上 bf16 的识别以免假警报。
    # `fp8_to_bf16` modules (engram.wkv) are converted but produce NO
    # `.weight_packed` (they emit a dense bf16 `.weight` instead), so they must
    # be counted separately -- otherwise a complete checkpoint looks short by
    # exactly their number.
    packed_n = sum(1 for k in index if k.endswith(".weight_packed"))
    # ★ 修正：原先把"bf16 化的模块"硬编码成只认 `engram.wkv`（历史上只有它一个），
    #   于是 --shared-experts-bf16/--attn-bf16/--attn-merged-bf16 新增的 352 个
    #   fp8_to_bf16 模块一律不被计数 ⇒ 完整产物反而报 `ERROR: 47235 vs 47587`，
    #   差额恰好等于 bf16 模块数（本会话实测：47587-47235=352）。
    #   正确做法是用权威判据 classify()，**不要靠猜名字**。
    bf16_n = sum(1 for k in index
                 if k.endswith(".weight") and classify(k[: -len(".weight")]) == "fp8_to_bf16")
    global_converted = packed_n + bf16_n
    this_run = stats["fp4_expert"] + stats["fp8_block"] + stats["fp8_to_bf16"]
    print(f"[convert] counters      : {stats}")
    print(f"[convert] converted     : {this_run} this run, {global_converted} in checkpoint "
          f"(expected {expected} for this selection)")
    # Only a full sweep can assert the checkpoint-wide total; a targeted subset
    # legitimately differs (and the checkpoint may still be missing shards).
    if not args.shards and global_converted != expected:
        raise SystemExit(
            f"[convert] ERROR: checkpoint has {global_converted} converted modules "
            f"but {expected} were expected — re-run (optionally with --skip-existing) "
            f"to finish the missing shards")
    print(f"[convert] DONE in {(time.time()-t0)/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
