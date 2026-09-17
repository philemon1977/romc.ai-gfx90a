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
                   engram.wkv, mtp.main_proj, ffn.shared_experts.{w1,w2,w3},
                   engram.embed  ->  F8_E4M3 + ``.scale`` F8_E8M0 at **32x32**
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
    """Classify a canonical module name (no ``.weight``/``.scale`` suffix).

    Returns ``fp4_expert``, ``fp8_block`` or ``keep``.
    """
    # routed MoE experts: FP4 (1x32) -> int4 (1x32). Size-neutral.
    if re.search(r"\.ffn\.experts\.\d+\.(w1|w2|w3)$", module):
        return "fp4_expert"
    # attention projections (MLA q/kv/o + indexer q)
    if re.search(r"(^|\.)attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$", module):
        return "fp8_block"
    if re.search(r"(^|\.)attn\.indexer\.wq_b$", module):
        return "fp8_block"
    if re.search(r"\.ffn\.shared_experts\.(w1|w2|w3)$", module):
        return "fp8_block"
    if module.endswith(".main_proj"):
        return "fp8_block"
    # ``engram.embed`` is a ParallelEngramEmbedding: vLLM gives it
    # ``embed_tokens.weight`` + ``embed_tokens.weight_scale_inv``, so the source
    # FP8 pairs load directly -> keep byte-identical.
    if module.endswith(".engram.embed"):
        return "keep"
    # ``engram.wkv`` IS a ReplicatedLinear, but vLLM builds it UNQUANTIZED
    # (bf16 ``weight`` only — verified: its params_dict holds
    # ``engram.wkv.weight`` and nothing else, and the DSV4 mapper rewrites our
    # ``.scale`` into ``weight_scale_inv``, which then exists nowhere).
    # Shipping the source FP8 bytes for a bf16 parameter would copy raw E4M3
    # bytes into a bf16 tensor, i.e. numerically wrong, so dequantize to bf16
    # and drop the scale.
    if module.endswith(".engram.wkv"):
        return "fp8_to_bf16"
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
    hi = (b >> 4) & 0x0F                       # element 2j
    lo = b & 0x0F                              # element 2j+1
    q = torch.stack((hi, lo), dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)
    lut = torch.tensor(_FP4_LUT, dtype=torch.float32, device=packed.device)
    val = lut[(q & 0x07).long()]
    val = torch.where((q & 0x08) != 0, -val, val)
    s = e8m0_to_f32(scale).repeat_interleave(32, dim=1)
    assert s.shape == val.shape, (s.shape, val.shape)
    return (val * s).to(dtype)


def dequant_fp8_block(w: torch.Tensor, scale: torch.Tensor,
                      dtype=torch.bfloat16) -> torch.Tensor:
    """F8_E4M3 + blockwise E8M0 scale -> dense (N, K)."""
    assert w.dim() == 2 and scale.dim() == 2, (w.shape, scale.shape)
    n, k = w.shape
    br, bc = n // scale.shape[0], k // scale.shape[1]
    assert br * scale.shape[0] == n and bc * scale.shape[1] == k, (w.shape, scale.shape)
    s = e8m0_to_f32(scale).repeat_interleave(br, dim=0).repeat_interleave(bc, dim=1)
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
            sd = s[r0:r1].to(dev)
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
    args = ap.parse_args()

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
            if nm.endswith(".weight"):
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
    print(f"[convert] policy : fp4_expert=FP4(1x32)->int4(g32)  "
          f"fp8_block=FP8(32x32)->int4(g32)  keep=byte-identical  "
          f"engram.wkv=FP8->bf16  engram.embed=kept")
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
                out_tensors.update(convert_module(module, w, s, kind, dev,
                                                  torch.bfloat16, row_chunk,
                                                  min_free_gib=args.min_free_gib,
                                                  wait_max_s=0.0 if args.no_wait_for_gpu else args.wait_max_s,
                                                  wait_poll_s=args.wait_poll_s))
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
        "lm_head", "head", "embed", "norm",
        "*mlp.gate", "*ffn.gate", "*shared_expert_gate*",
        "vision.*", "aligner.*",
        "*attn.compressor.*", "*attn.indexer.wk", "*attn.indexer.weights_proj",
        "*attn_norm", "*ffn_norm", "*kv_norm", "*q_norm", "*k_norm",
        "*confidence_head*", "*markov_head*",
        "*hc_attn*", "*hc_ffn*", "*attn_sink*",
    ]
    ignore += ["*engram*"]
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
    global_converted = sum(1 for k in index if k.endswith(".weight_packed"))
    this_run = stats["fp4_expert"] + stats["fp8_block"]
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
