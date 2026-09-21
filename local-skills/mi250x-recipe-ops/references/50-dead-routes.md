# 死路与负结论（负结论也是资产）

> 每条都注明被否证时的 scope。**在一个 scope 上判死 ≠ 在另一个 scope 上也死。**

<!-- ── 搬运自 SKILL.md L145-253 ── -->
> **跨层 · 负结论** — 每条都注明它是在哪个 scope 上被否证的；**换 scope 不等于结论仍成立**。

## Dead routes (negative results are assets)

Do not spend boot time re-testing what is recorded in `data/*.json` under
`status: dead` / `what_failed`; plus **live-verified 2026-09-15**:
Ornith-1.5-397B-**FP8** on vLLM/gfx90a dies at engine init (`torch._scaled_mm`
requires MI300+/CC>=8.9) after a fully healthy 389.6 GiB weight load —
this model on this host = llama.cpp Q8_0 (8110) only — e.g. AITER pybind11 jit tree without the ABI
patch, ROCm 10 AITER path on gfx90a, vLLM TP8 QuickReduce defaults (see
knobs), prefix-cache assumptions on 27B BF16 MTP trio. Each entry cites the
disproving recipe.

**Why that FP8 verdict is a silicon gate, not a tuning miss** (vendor
corroboration, `data/vendor_references.json`): the official vLLM recipe for the
base model `Qwen/Qwen3.5-397B-A17B` lists AMD support as
`{MI300X, MI325X, MI355X}` only — MI250X is absent from the whole matrix, and
the entire official AMD lane is the FP8 checkpoint with FP8 GEMM. gfx90a has no
FP8 matrix core, so stop looking for a config that makes stock FP8 work here.
What *does* transfer from that recipe: `--language-model-only` (text-only, frees
HBM per die), `VLLM_USE_DEEP_GEMM=0` + `VLLM_DEEP_GEMM_WARMUP=skip`,
`--trust-remote-code`, `--enable-prefix-caching`, and its hybrid GDN+Mamba
pitfall `assert num_cache_lines >= batch` → lower
`--max-cudagraph-capture-size` (default 512). The local gfx90a FP8→BF16
dequant-emulation kernel is the only route that has loaded this checkpoint's
weights on vLLM here (48.27 GiB/die, TP8, then hybrid cache page alignment and
graph capture), and it is now measured end to end on the official InferenceX
client. Single stream (CONC=1): **21.6 tok/s @1k context** (TPOT 45.4 ms),
5.4 @32k, 1.81 @128k, 1.03 @240k; CONC=8 -> 60.8 and CONC=16 -> 138.1 and
CONC=32 -> 236.7 tok/s aggregate.

**The long-context wall is a kernel, not the silicon.** Decode TPOT grows
*linearly* with context: `TPOT ~= 45.4 ms + 3.97 ms x (ctx/1024)`, fitted across
32k/128k/240k within 1.6 ms, extrapolating `262,144 -> 0.94 tok/s`. At 128k that
is ~2.0 GiB/die of KV read per token in 0.509 s = **0.30% of HBM peak**, i.e. the
`ROCM_ATTN` paged-attention path (no split-KV/flash-decoding), not a hardware
limit. Turning `VLLM_ATTENTION_BACKEND` on this route is the first lever to try,
scored on `TPOT@128k = 553.9 ms`.

**MTP speculation works on this route and the win grows with context** (the
checkpoint's `mtp.*` tensors are BF16 -- `quantization_config.ignore` carries
`re:.*mtp\..*` -- so they never touch the FP8 path; method `qwen3_5_mtp`, k=2):
2.14x @1k -> 3.40x @240k single stream, **but zero gain at CONC=16**
(138.1 -> 133.6 tok/s aggregate). Quote acceptance with its content: official
synthetic `random`+`--ignore-eos`+greedy inflates it (88.4% rate, mean length
2.72, 27% of windows saturated at the k=2 ceiling), while real repo review turns
give 75.3% / 2.51 and a **1.88x** speedup (19.05 -> 35.86 tok/s).

**Capacity, priced at 16.0 KiB/token/die** (measured: 6.9 GiB -> 452,748 tokens):
1x256k fits, 16x27k is the envelope, 16x32k is ~0.9 GiB/die short (use
`--gpu-memory-utilization 0.98` or FP8 KV), 1x1M does not fit while the emulation
kernel pins BF16 weights at 48.27 GiB/die -- keeping weights 8-bit resident with
tile-load dequant frees ~23.9 GiB/die and is what unlocks 1M. `--max-num-seqs
256 -> 16` triples the pool (154,624 -> 452,748 tokens): GDN state is allocated
per slot, so slot count is a memory knob, not just a scheduler knob. vLLM refuses
`max_model_len > max_position_embeddings` (262,144) without
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.

The lane is preset-launchable and the patch is a durable artifact:
`hyperloom/presets/ornith-agent-longctx/` (`launch.sh`, `README.md`, private asset
root with `timeout_seconds: 1800`) and
`hyperloom/patches/fp8-w8a8-emulation-gfx90a/` (overlay + `bin/vllm-emulation` +
398-line diff vs vLLM 0.28.0). To get a patched framework into the *benchmark
server* use `--extra-env VLLM_BIN=<wrapper>`: `PYTHONPATH` is blocked on every
untrusted channel (KB `extra_envs`, variant envs, `--extra-env`,
`--reference-script` exports), and the first-party alternative is a provisioned
`StackRuntime` -> `pythonpath_prefix` -> `apply_runtime_override`. Still no
optimizer sealed baseline on this route (`baseline_tput=0.0`), so its
`best_throughput` stays 0.0 and the llama.cpp Q8_0 arm's 47.28 tok/s (ngram+MTP)
remains a different, non-comparable measurement protocol.

**Profile a step before running more A/B rounds.** Six end-to-end rounds on Ornith
could only characterise speculation's long-context cost ("one O(ctx) term per
speculative step, independent of k": 435.1 ms/step at k=1 vs 451.6 ms at k=2 at 128k,
against 42.7 ms with split-KV and no speculation) and produced two falsified
attributions. One profiled window named it: with `max_query_len > 1`,
`chunked_prefill_paged_decode` runs the Triton prefix-prefill over the *whole*
context for every layer and then falls through to the decode branch, where split-KV
overwrites the same output rows -- 288 calls x 26.0 ms in a 17-step 145k-context
window, 78.9% of the step, pure waste. How to get that window on a ROCm box:
`--profiler-config.profiler torch --profiler-config.torch_profiler_dir <dir>` plus
`POST /start_profile` / `POST /stop_profile`; each worker writes
`profiler_out_<rank>.txt` (a `key_averages().table()` per-kernel CUDA-time ranking).
`rocprofv3 --attach` is *not* a substitute here: on this host it prints `:: success`
and writes nothing. Send the same long prompt twice with prefix caching on so the
profiled request is decode-only, otherwise the one-time prefill (~15 x 25 ms of that
same kernel) pollutes the attribution.

**Before hand-writing a gfx90a kernel, check for the one that ships disabled.**
On Ornith-1.5-397B-FP8 the long-context decode wall (`TPOT ~= 45.4 ms + 3.97 ms x
ctx/1024`, i.e. 0.30% of HBM peak because the fallback is ONE workgroup per
sequence+KV head) is fixed by `VLLM_ROCM_SPLITKV_PA=1`: a MI250X split-KV
(flash-decoding) decode kernel already present in the wheel at
`vllm/v1/attention/ops/rocm_splitkv_pa.py`, dispatched one line before the serial
kernel, off by default. Measured single stream, no speculation: 128k TPOT
553.9 -> **42.7 ms** (x13.0), 240k 967.9 -> 44.7 ms (x21.7), slope 3.97 -> 0.025 ms
per 1k tokens. Two traps that cost real time: (a) `head_dim=256` fails the native
ROCm paged-attention gate (`platforms/rocm.py:403` allows 64/128) and the hybrid
`block_size=528` is not a power of two, which is *why* it lands on the serial path;
(b) widening the kernel to serve speculative steps (`VLLM_ROCM_SPLITKV_PA_MAX_Q=3`,
patch + tests in `hyperloom/kernels/gfx90a_flash_decode/`) is *necessary but not
sufficient*: the scratch budget story was a red herring (raising
`VLLM_ROCM_SPLITKV_PA_MAX_SCRATCH_MIB`/`_MAX_TOTAL_MIB` changed nothing once
`takeover` was non-zero) -- the multi-row batch never reaches split-KV *first*,
because the redundant Triton prefix-prefill pass above it dominates; see the
`multirow-skip-triton-prefill.patch` dispatch fix and the profiling note above.
`VLLM_ATTENTION_BACKEND=TRITON_ATTN` was measured as ~21% *slower* at 128k: do not
reach for it. And a microbench for these kernels must hand them a block table as
wide as a live server's (`max_model_len/block_size`); a minimal one reads past the
table and raises `Memory access fault by GPU`, which looks exactly like a kernel bug.


<!-- ── 搬运自 SKILL.md L474-581 ── -->
> **T2 · 模型/量化层（DSV4.1 线）** — FP4 nibble 序、indexer 缓存布局、cudagraph 事实、engram 死路、判据设计、取证法。

## 本会话新增（2026-09-20：DSV4.1-Flash CT-int4 根因、稀疏 indexer 内核与 engram 死路）

来源：`DeepSeek-V4.1-Flash-CT-Int4-W4A16` 线（端口 8119 / 补丁树 `ct_w4a16_dsv41_n0918`）。
完整记录：`$AI/docs/DeepSeek-V4.1-Flash-CT-INT4-W4A16-转换记录-2026-09-18.md` §4.34–§4.39。
以下四条此前在本技能 **0 命中**。

### A. ★ FP4 nibble 序 bug 类（转换侧，最贵的一课）

- **症状**：量化后的模型"流畅但退化"——事实召回 0/6、复读吸引子；而同一批源文件走 vLLM 自带 loader 的
  官方编码臂**健康**。查注意力/engram/内核/镜像版本全部无效。
- **根因**：源 MXFP4 打包约定是「**低 nibble = 偶数元素**」（权威两处：模型目录自带
  `inference/convert.py::cast_e2m1fn_to_e4m3fn`；运行时 `_unpack_gptq_int32_to_signed_int4`），
  而我们的转换器写成 `hi` = 偶数元素 ⇒ 47,232 个专家张量全是源行的「**相邻元素对互换**」版。
- **判据（可复用）**：源 vs 仓逐元素相关——修复前 corr **+0.05**，对齐"相邻对互换"后 corr **+0.997**、
  relL2 0.065（≈4bit 重量化噪声量级）⇒ 结论是"只差一个相邻对排列"。
  工具：`quark-int8/audit_fp4_nibble_order.py <src> <ours>`。
- **修复**：字节内 nibble 互换（纯置换、无损、**自逆**；组 32 元素=16 字节，不跨字节 ⇒ scale 仍有效）。
  工具：`quark-int8/fix_fp4_nibble_order.py <repo> --measure|--apply`；守门测试
  `quark-int8/fix_fp4_nibble_order_test.py`（5 项，含"已修过再跑会自逆回退"这道闸）。
- ⚠️ **工具地雷**：无 journal 的老仓无法判断是否已修；先跑 audit（corr≈0.99 ⇒ 已修）再决定，别直接 --apply。
- **方法论（比 bug 更重要）**：此前的"转换保真度"检查是**闭环自证**——`cmp_layer_weights.py` 从
  `verify_ct_int4` import 了**同一个（错的）读取器**去读源文件，两边同错 ⇒ 全绿。
  **规则：自己写的转换，永远不能用自己写的读取器验收**；必须用厂商代码/运行时自己的解包实现当第三方尺子。

### B. indexer 缓存布局：技能里"默认不可信"的**原因**（补充）

技能第三轮记了「`DSV41_IDX_AITER_KERNEL=1` 才可信、默认回退不可信、+69.3%」，但没写为什么。三条独立证据：

1. 写入端 `indexer_k_quant_and_cache_triton`：`layout = "NORMAL" if block_size == 1 else "SHUFFLE"`；
   本机 `DeepseekV4IndexerCache` 用 `cache_config.block_size`，**实测 = 32**（不是 64）。
2. 上游自己的读取器 `cp_gather_indexer_k_quant_cache_triton` **也按 SHUFFLE** 处理 ⇒ SHUFFLE 是真实约定。
3. 写入→读回对比原值：行主序 corr **+0.014**（垃圾）vs SHUFFLE 反解 corr **+0.989** ✓
   （`quark-int8/idx_layout_probe.py`）。

⇒ 上游 torch 回退 `fp8_paged_mqa_logits_torch` 按行主序读：**把 scale 区字节当 fp8 值**读 ⇒ logits 与真值差
~1000×（真值量级 0.008 vs 回退 10.03）⇒ 候选块/top-k 选到非法位置 ⇒ 下游越界 ⇒ **worker 静默硬崩**
（无 Python traceback）；它还含 `.item()` ⇒ 图捕获必报 `hipErrorStreamCaptureUnsupported`。
**这就是"长上下文崩"与"FULL 图捕获不可能"的共同根因。**

- 正确读法（页内）：值区 `off(t,d) = (t//16)*16*D + (t%16)*16 + (d//16)*256 + (d%16)`，
  scale 区 float 下标 `= BS*D/4 + t`；**16×16 tile 常量与写入端默认值耦合**，上游改了要同步。
- 自研内核两个真 bug（已修+已回归）：① q 是 `[B, next_n, H, D]`，必须用 batch/step **两个** stride；
  用 `row*stride(0)` 在 `next_n>1` 时读错行（**稀疏 MLA warmup 的 mixed tokens=16 就踩这个**，
  且 `next_n=1` 时两种写法重合、看不出来）。② 页号需钳制。守卫：`q.dim()!=4` 与 `BS%16/D%16` 显式报错。
- **验证法**：一律用**生产写入端**造缓存（`indexer_k_quant_and_cache_triton`），以"原始 k/原始值"为真值三方对拍；
  `quark-int8/idx_kernel_verify.py`（覆盖 next_n=1/3 + 图捕获）。禁用自造布局——我因此漏掉过两个 bug。

### C. cudagraph 本机运维事实（补技能）

- **`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 默认开**：吃掉约 10 GiB/rank 预算
  （日志原话：`util=0.93 等效 0.7769`）⇒ 紧的时候必须 `=0`。
- FULL 图捕获**要求 indexer 路径无 `.item()`**：走回退必失败；`DSV41_IDX_AITER_KERNEL=1` 后 **FULL 2/2 通过**（实测）。
- 显存账：engram 驻显存时权重 51.7 GiB/rank，叠加图池超过 util 0.97 预算 ⇒
  `Available KV cache memory: -3.23 GiB`；需 `=0` 关预估 + 缩小 `MAX_MODEL_LEN`/`MAX_CUDAGRAPH_CAPTURE_SIZE`。
- **未决**：稀疏 MLA warmup 的第二轮捕获（11 张图）处 **Segfault**（`!!!!!!! Segfault encountered !!!!!!!`），
  与内核是否相关尚未做单变量区分（`DSV41_IDX_AITER_KERNEL=0/1` 同配置对比待做）。
- `--cpu-offload-params` 走 UVA 时**装载会走 HSA 慢路径**（perf 实测 libhsa 15% + libgomp 23%；
  装载 612 s → >1200 s）。

### D. engram 表进主机 RAM = 死路（三段证据，别再花时间）

1. **参数名整段匹配**：vLLM 卸载匹配是 `f".{param}." in f".{prefix}{name}."`；传 `engram.embed` 匹配不上
   `...engram.embed_tokens.weight` ⇒ **零个参数被卸载**（显存不掉、RAM 不涨）。正确段名 `embed_tokens.weight`、
   `embed_tokens.weight_scale_inv`（不能只写 `weight`，会匹配全模型的 Linear）。预算语义：`--cpu-offload-gb` **每 rank**。
2. **释放不还显存**：自研"装载后搬"（装载 638 s ✓ 正常）后 `empty_cache` **释放 0.00 GiB**
   （权重与表落在同一个 `expandable_segments` 大段里，段未腾空就不还给驱动）⇒ 算得 `Available KV: -3.23 GiB`。
3. **强行释放会 GPU Hang**：改 `PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8` 后出现
   `HW Exception by GPU node-2 … reason: GPU Hang`。
   （顺带：启动器曾把 `PYTORCH_HIP_ALLOC_CONF` 写死，已改成可覆盖 `${VAR:-default}`。）

⇒ 可行但**不划算**的替代：`DSV41_ENG_HOST_PREALLOC=1`（表一开始就建在 pinned 主机内存，显存 41.0 GiB/rank ✓），
但装载要多花 10–20 分钟填 97 GB pinned 页。**结论：不采用**——用 `DSV41_ENG_SKIP=1` 或让表驻显存。

### E. 判据设计与脚本自伤（补充既有清单）

- **判据必须对置换敏感**：我用"每位置 `max|v|` 对比 `448*scale`"判布局——**对 d 取 max 是置换不变的**，
  两种读法给出完全相同的比值（都是 796.44）⇒ 该判据零区分度。"**写入→读回对比原值**"才一刀切开。
- **臂脚本自伤（一天三次同类）**：① `pkill`/`kill` 打到自己 ⇒ 要**祖先链豁免**
  （从 `/proc/<pid>/stat` 逐级取 PPid）；② 上一轮臂脚本用**共享 PID 文件** + `docker rm -f` 把新臂容器
  和进程一起打掉 ⇒ 收尾前必须 `owns_container`（比对 `docker inspect .State.StartedAt` 与本轮起始时间）；
  ③ **启动器 env 白名单漏项** ⇒ 关键开关静默不生效、白等一轮（`perf_cg_arm.sh` 当时还把
  `DSV41_IDX_AITER_KERNEL` 自己 `unset` 了，两个原因叠加）。
  ⇒ 臂脚本必须在起臂后 **30 秒内自检容器内 env**（`docker exec … env | grep '^DSV41_'`），不一致立即中止。
- **静默期不是卡死**：专家 repack（47,232 张量）**10–20 分钟零日志**、CPU 满转、零缺页、零磁盘读。
  判死前先看 CPU 增量与缺页增量；我因误判"卡死"反复杀掉正常臂，是本会话最大的时间浪费。
- 改动后跑 `bash quark-int8/scripts_local/check_my_code.sh`：补丁文件 `py_compile`（能抓 `ast.parse`
  抓不到的 `global` 顺序/未定义名）+ 工具编译 + 全部 shell `bash -n` + CPU-only 守门测试。

### F. 两个可复用取证法

- **是否真在用矩阵核（MFMA）**：用 gfx90a target 编译同形状 `tl.dot`，查生成 `amdgcn` 汇编有无 `v_mfma_*`、
  有无 `v_fma`（`quark-int8/mfma_evidence.py`）。实测 int4 路径：
  `v_mfma_f32_16x16x16bf16_1k`、**零 FMA**；CDNA2 无 fp8 矩阵核（aiter 的 fp8 内核在 gfx90a 上编译直接失败
  `Unsupported lhs dtype fp8e4nv`）。
- **TTFT/TPS 口径**：TTFT 用 `max_tokens=1` 墙钟近似；**TPS 必须 `ignore_eos=True` +
  `min_tokens=max_tokens`**，否则短答立刻 EOS、解码 TPS 变 nan（踩过）。工具 `quark-int8/perf_bench.py`；
  事实召回探针已加"期望子串"表（`fact_recall_probe.py` 输出 PASS/FAIL 与命中率），别再靠肉眼判质量。

### G. 与本技能既有条目的关系

- 技能第三轮记的「`DSV41_IDX_AITER_KERNEL=1` +69.3%（conc32）」挂的**正是本会话改的那份文件**
  （`ct_w4a16_dsv41_n0918/tree/v1/attention/ops/rocm_aiter_mla_sparse.py`，GLM-5.3 int4 的 8121 launcher 同源）
  ⇒ 该条既是速度证据，也**反证内核端到端可用**；但 DSV4.1 线路上的长上下文崩溃（§4.39）仍未闭。
- ⚠️ 同名文件在工作区有 **4 份**，SHUFFLE 内核只在 ops 那一份里（`backends/mla/` 与旧树 `ct_w4a16_dsv41/`
  均为 0）⇒ 换补丁树/换线路前先核对：`grep -c USE_SHUFFLE <file>`。

---

