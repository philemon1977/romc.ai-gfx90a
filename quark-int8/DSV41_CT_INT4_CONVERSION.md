# DSV4.1-Flash → compressed-tensors INT4 W4A16: conversion record

Date: 2026-09-18 · host: 8× AMD Instinct MI250X (gfx90a, CDNA2) · converter:
`quark-int8/convert_dsv41_ct_int4.py`

- source: `/mnt/stripe-3mix-3t2/models/deepseek-ai/DeepSeek-V4.1-Flash` (475.3 GiB, 48 shards)
- output: `/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16` (521.4 GiB)
- goal: reach the ROCm `TritonW4A16LinearKernel` / WNA16-TRITON MoE path instead of the
  MXFP4 emulation that made the Ornith int4 arm slow.

## 1. Why not Quark (the original request)

| asked | outcome |
|---|---|
| Quark INT8 attn | vLLM's `QuarkConfig` has **no int4 weight-only scheme** (`NotImplementedError: No quark compatible scheme was found` for int4/uint4 per_group\|per_channel — measured in `INT4_CT_PLAN.md`); and `vllm/models/deepseek_v4_1/quant_config.py:override_quantization_method` claims `model_type: deepseek_v41` only for `fp8` or Quark-**MXFP4**-OCP, so a Quark int8 export cannot load DSV4's MLA/engram linears. |
| Quark INT4 experts+attn | same `NotImplementedError`. Quark was abandoned for Ornith int4 for exactly this reason (shipped as compressed-tensors). |
| aiter CK INT4 GEMM | at S1/4 in `MI250X-AITER-INT4-内核复核-2026-09-17.md` §6.2 (S2 int4 MoE instances, S3 stage2 fp32 reduce, S4 vLLM W4A16-MoE→aiter wiring all undone). Not reachable by a conversion. |

Therefore: **compressed-tensors `pack-quantized` int4 W4A16.**

## 2. Source layout (measured, all 48 shards)

| group | format | size |
|---|---|---|
| routed experts `layers.N.ffn.experts.E.{w1,w2,w3}` | I8 packed FP4 + `F8_E8M0` scale, 1 row × 16 packed bytes = **1×32 fp4** | 275.7 GiB (58%) |
| attention (wq_a/wq_b/wkv/wo_a/wo_b, indexer.wq_b), ffn.shared_experts.{w1,w2,w3}, mtp.main_proj, engram.wkv | `F8_E4M3` + `F8_E8M0` scale at **32×32** blocks | 5.7 GiB |
| engram.embed tables (`layers.{1,14}`) | `F8_E4M3` + E8M0 at 1×32 | **189.1 GiB (40%)** |
| norms / routers / embeddings / head / visual | BF16 | ~5 GiB |

Note `.weight` for experts is `I8` holding 2 FP4 nibbles/byte and `.scale` is
E8M0 at 1 scale per 32 FP4 elements — the same convention Quark's own
`_is_mxfp4_source_pattern` recognises for DSV4.

## 3. Conversion policy

| category | action | count |
|---|---|---|
| routed experts (FP4 1×32) | dequant → int4 g32 symmetric, fp32 scale | 47,232 |
| attention + shared experts + main_proj + indexer (FP8 32×32) | dequant → int4 g32 symmetric | 353 |
| engram.embed / engram.wkv | **kept verbatim** (they are Embedding modules; opt in with `--quantize-engram`) | 4 |
| everything else | byte-identical pass-through | 907 |

`group_size=32` is deliberate: it matches the source's native 1×32 blocks and
hits vLLM's dedicated `kInt4Static32GroupScale` WNA16 case.

## 4. Verification actually performed

| check | result |
|---|---|
| `pack_int4_g32` vs `PackedQuantizationCompressor.compress` | **byte-identical** `weight_packed` |
| FP4 dequant sanity | reproduces exact E2M1×2⁻⁵ grid (max\|w\| = 6×2⁻⁵ = 0.125) |
| CT config parse | `pack-quantized`; `_get_scheme_from_parts` → **`CompressedTensorsWNA16`** |
| **kernel actually selected** | `choose_mp_linear_kernel(int4, g32, sym, bf16 acts)` → **`TritonW4A16LinearKernel`** |
| MoE backend | `_get_priority_backends()` → **TRITON** (RDNA3/FLASHINFER/MARLIN arch-gated, EMULATION last) |
| full checkpoint headers (48 shards) | **47,585 / 47,585 modules**, missing=0 extra=0 FAILURES=0 |
| numeric spot checks, experts (4 shards) | relRMSE ≈ 0.100, cos ≈ 0.995 |
| numeric spot checks, attention (17 modules incl. `wo_a` bmm, indexer, mtp) | relRMSE ≈ 0.098–0.101, cos ≈ 0.995–1.009 |
| name/shape contract per module | `weight_packed` int32 [N,K/8], `weight_scale` fp32 [N,K/32], `weight_shape` int64 [2] with **logical** K |
| **fused `wq_a`+`wkv` int4 loading** (`test_fused_wqa_wkv_load.py`) | **PASS** — via vLLM's real `load_merged_column_weight`: fused buffer == `concat(wq_a, wkv)` for both `weight_packed` (bit-exact) and `weight_scale`; round-trip relRMSE 0.0985, cos 0.9964 |
| **real DSV4.1 attention layer accepts the tensors** (`test_real_layer_load.py`) | **PASS** — built with the converted `quantization_config`: the layer's own quantized params are created with exactly our shapes (`fused_wqa_wkv` 1792×640 + scale 1792×160, `wq_b` 32768×160, `wo_b` 5120×4096); `wq_a`→shard 0 (offset 0) and `wkv`→shard 1 (offset 1280) both load |

### 装载语义（读源码 + 实测得到的两条硬约束）

1. 合并列装载的 `shard_offset`/`shard_size` 必须用**未打包单位**传入：
   `ModelWeightParameter.load_merged_column_weight` 内部用
   `_adjust_shard_indexes_for_packing` 除以 `packed_factor`（int4 即 8）。
   我最初按打包单位调用，偏移算成 160（应为 0），被测试抓出来。
2. `TritonW4A16LinearKernel` 只实现 fp16/bf16 激活：构造量化层时
   `params_dtype` 必须是 bf16 `LinearBase` 默认取 `torch.get_default_dtype()`
   （fp32），会让内核选择直接报
   `TritonW4A16LinearKernel cannot implement due to: Only float16/bfloat16 activations are supported`。
   真实 engine 会传对；这也是启动脚本必须显式 `--dtype bfloat16` 的原因。

### 整机 TP8 起服实测（2026-09-18，vLLM nightly）

起服**成功走到了权重全部装上 GPU**，两条内核证据都在：

```
Using TritonW4A16LinearKernel for CompressedTensorsWNA16      (8/8 rank)
Using CompressedTensorsWNA16MoEMethod
Using 'TRITON' WNA16 MoE backend.
```

- 权重装载：48/48 分片，26 秒。
- 显存：每卡 63.1–63.3 / 64 GiB —— **521 GiB 的 checkpoint 恰好装进 512 GiB 显存**（8×64 GiB）。
- GEMV 泛化补丁生效：`[MI250_MOE_GEMV] kernel module = mi250_moe_gemv_gs (generic)`。

### 起服路上踩掉的三个坑（都是本文档前面结论的直接后果）

1. **`is_layer_skipped` 默认 `match_mode="exact"`，不支持通配符**，
   而 `should_ignore_layer(..., use_fnmatch=False)` 让 CT 侧的 `ignore` 里所有 `*` 规则
   全部静默失效（只有无通配符的 `lm_head` 命中）。于是 `engram.wkv` 被当成量化层，
   而 DSV4.1 的装载器要的是未量化 `.weight` ⇒ `KeyError: 'layers.1.engram.wkv.weight'`。
   修法：给 CT 打 `use_fnmatch=True` 一行补丁（上游那行上面正好挂着
   `# TODO (@kylesayrs): support ignore module names with ct matching utils`）。
   Ornith 的 config 之所以能工作，是因为它把 2092 条**精确全名**都枚举了。
2. **未量化的 FP8 层必须反量化成 bf16 落盘，不能原样保留 FP8。**
   DSV4 的 mapper 会把 `.scale` 改写成 `.weight_scale_inv`，而 vLLM 为未量化层只注册
   `weight` ⇒ 名字对不上；**就算对上了也是错的**：FP8 的原始字节被 `.copy_()` 进 bf16
   参数，数值全废。故 `engram.wkv` 走 `fp8_to_bf16`（`engram.embed` 因为有
   `embed_tokens.weight_scale_inv` 参数，保持 FP8 正确）。
3. **`weight_scale` 必须按 bf16 落盘。** vLLM 用 `params_dtype`（本模型 bf16）建
   `weight_scale` 参数，而原先写的是 fp32 ⇒ 47,585 个 scale、约 175 亿元素在加载时
   全部要做一次 dtype 转换，这正是启动耗时几十分钟、内存从 123 涨到 230 GiB 的来源。
   bf16 的代价实测 **0.389%** 最大相对误差，而 int4 自身量化误差约 **19.8%** —— 可忽略。

### 启动耗时：一个尚未消除的已知代价

vLLM 的 Triton W4A16 kernel 要 **K-packed** 布局（`qweight [K, N//8]`），
而 compressed-tensors `pack-quantized` 是 **N-packed**（`[N, K//8]`）；两者不是
transpose 关系，需 unpack→transpose→repack。源码自己写着
"This is done CPU-side at load time (one-time cost)"（实际张量在 CUDA 上，
`_transform_param` 只做转发，瓶颈是**每个专家层走一遍 Python + 28 万次小 kernel 发射**）。
对 47,232 个专家层做这件事需要几十分钟，且**不落盘缓存，每次重启都要重付**。

- 方案 C（批量 repack）的数值前提已证明：`test_batched_repack.py` 显示批量版本与
  vLLM 逐层版本 **逐位一致**（12 个专家、两种形状全 True）。
- 是否落地取决于消除上面第 3 条后启动还剩多久。

### 仍未完成

- **吞吐未测**（t/s、step ms）——这才是本次转换的目的。测量脚本已就绪：
  `bench_serve.sh`（分段计时：weight-load / init→ready / 首 token）+ `bench_infer.py`。
- 无质量对拍（NLL/ppl 对比 FP8/FP4 源）。
- 走的是 **Triton W4A16**，不是最初想要的 **aiter CK INT4 GEMM**（见 `MI250X-AITER-INT4-内核复核-2026-09-17.md` §6.2 S2–S4）。
  追加发现：项目**没有**为 gfx90a 实现 `--moe-backend aiter` —— `aiter_patch/sitecustomize.py`
  自述只放开 **int8 linear**，fused MoE 等仍走上游门控；gfx90a 上真正可用的 MoE 优化是
  `moe_gemv_patch/` 自研 Triton GEMV（非 aiter，实测 +62%），本次已用泛化版
  `mi250_moe_gemv_gs.py` 接上（原版 `GROUP=128` 硬编码，对 gs=32 会静默回退上游）。


### 启动入口（按本机 launcher 规范）

**唯一入口**：`/home/qiba/ai/models/deepseek-ai/launcher/deepseek_v4.1_flash_ctint4w4a16_vllm_rocmnightly_256k_8119_dsv41_mi250dx8.sh`

- `setsid` 后台起服、PID 文件 `${AI_HOME}/logs/dsv41ctint4-8119.pid`、
  日志 `${AI_HOME}/logs/dsv41ctint4/server-8119-<ts>.log` + `server-8119.current` 软链。
- **两个补丁的挂载逻辑全部内联在脚本里**，并有存在性 + 内容预检：
  ① `diag_ovl/compressed_tensors/compressed_tensors.py`（`use_fnmatch=True`，
     脚本会 `grep -q use_fnmatch=True` 校验，缺了直接拒启）；
  ② `moe_gemv_patch` → `/patches/moe_gemv`（PYTHONPATH 前置）。
- 预检还包括：模型/config/index 存在、`DSV_MOE_BACKEND=aiter` 直接拒（gfx90a 会 raise）、
  8 张卡空闲 ≥ `VRAM_FREE_MIN_GIB`（默认 40 GiB，他方占卡则快速失败）。
- 早期那份 `quark-int8/serve_dsv41_ct_int4.sh` **已删除**：它没有 use_fnmatch 挂载，
  留着会有人踩回 `KeyError: 'layers.1.engram.wkv.weight'`。

分段计时 + 冒烟 + 单流 t/s：`quark-int8/bench_serve.sh`（内部调用上面的 launcher，
保证补丁只在 launcher 里挂一次）+ `quark-int8/bench_infer.py`。

期望日志：`Using TritonW4A16LinearKernel for CompressedTensorsWNA16`（8/8 rank）、
`Using 'TRITON' WNA16 MoE backend.`、`kernel module = mi250_moe_gemv_gs`。

## 5. Cost / size

- weights 475.3 → **521.4 GiB (+9.7%)**; the converted portion grows +16.1%
  because int4 experts switch from 1-byte E8M0 scales to 4-byte fp32 group scales.
  The 94 GiB engram shards are unchanged.
- full run ≈ 75 min of GPU work across 31+17 shards, plus ~25 min for the two
  94 GiB engram shards. Peak host RAM 183/251 GiB (chunked pass-through added
  afterwards; see `get_passthrough`).

## 6. Operational notes

- The run is **resumable**: `SKIP_EXISTING=1 ./run_convert.sh`.
- Coexistence with other GPU tenants: `--min-free-gib` guard + `--wait-max-s`
  waiting + adaptive chunking + OOM-halving. A co-tenant starting mid-run is what
  caused the only failure (OOM), hence these.
- A first attempt under `--skip-existing` aborted **after** writing `config.json`
  and the merged index because the final assertion counted only this run's
  modules. Fixed: the check now counts `*.weight_packed` in the final index.
