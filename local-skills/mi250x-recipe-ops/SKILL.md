---
name: mi250x-recipe-ops
description: >-
  Operates the local 8x AMD MI250X (gfx90a / MI200) workstation using its
  verified recipe library: which serving arm runs on which port, how to
  start/stop/observe it, how to rebuild its environments (vLLM 0.28 / master,
  llama.cpp gfx90a, ROCm 7.2.4, wu1w INT8), how to apply/verify/revert its
  patches (QuickReduce, splitKV, DSA indexer, AIS/PLE, aiter), how to set its
  tuning knobs (MTP/spec depth, AIS, --fit off, TP split-mode), and which
  routes are proven dead and must not be re-attempted. Use when the user asks
  about "this machine's" models/ports/launchers/recipes, mentions ports 8101 or
  8103..8114/8203/8301/8302, asks to start/stop a local endpoint on MI250X
  hardware, or wants Hyperloom's local RecipeKB warm-start for these arms. Do
  not use for MI300X+ or Docker-based generic serving (see
  serving-llms-on-instinct).
allowed-tools: Bash, Read
---

# MI250X Workstation Recipe Ops

Single-machine operating manual for `/home/qiba/ai` (called `$AI` below): 8x
MI250X dies, ROCm 7.2.4, vLLM (0.28.0 + master) and llama.cpp arms, each with
a verified launcher under `models/*/launcher/*.sh`.

The authoritative sources are the recipe files themselves under
`$AI/docs/recipes/` (gated by `$AI/tools/audit_recipes.py`). The JSON files in
`data/` are machine-readable extracts for agents; if they disagree with the
recipe markdown or the launcher scripts, the scripts win — re-run
`scripts/build_skill_data.py` in the ROCm.AI workspace to refresh.

## Host facts (never re-derive)

- Repo root: `/home/qiba/ai` (all paths below are relative to it).
- 8x MI250X GPU-dies (gfx90a). VRAM usage per die in **bytes**:
  `paste <(seq 1 8) <(for c in /sys/class/drm/card{1..8}; do cat $c/device/mem_info_vram_used; done)` — idle ≈ 1.0e7.
- Power cap 560 W/module (`/etc/amdgpu-powercap.conf`); align cap before any
  cross-run comparison.
- Port registry + mutual exclusion: `config/ports.conf`. Never assume a port
  is free; probe: `nc -z 127.0.0.1 $P; ss -ltnp | grep ":$P "`.

## Step 0: pre-flight before starting anything

1. Check which dies are occupied (command above).
2. Check who holds kfd: `sudo -n fuser -v /dev/kfd 2>/dev/null || fuser -v /dev/kfd`.
3. Check target port + its PID file: `cat logs/*-$P.pid 2>/dev/null`.
4. Align/inspect power cap if the comparison depends on it.

## Start / stop / observe

- Start = run the arm's `entry` script from `data/arms.json` (it bakes in all
  fail-closed guardrails; do NOT re-implement its args).
- Stop = kill the session group of the PID file only:
  `kill -TERM -$(cat logs/<key>-<port>.pid)`.
  **NEVER** `pkill -f` and never feed `pgrep` output to `kill` — this host runs
  many concurrent instances and has killed a production endpoint that way.
- Observe: `docs/日志规范-2026-09-12.md` conventions; logs + PID under `logs/`.

## Performance measurement rules (three hard gates)

1. Warm up first — the first request pays Triton autotune/JIT (26 vs 62 t/s on
   Ornith). Never report the first number.
2. Cross-boot drift sigma ≈ 4.6% — any claim below ±2% from a single boot is
  invalid; never mix absolute values across boots/builds in one table.
3. If single-stream TPS < 20, stop immediately, record "性能不达标，止于 X t/s",
   do not fill in remaining dimensions.

## Serving arms (see `data/arms.json` for full config)

| Port | Arm | Engine | Status |
|---|---|---|---|
| 8101 | Qwen3.8-27B Fable-Distill BF16 TP2 (vLLM master) | vllm | active |
| 8103 | DeepSeek-V4-Flash-0731 284B Q8_K_XL | llama.cpp | active |
| 8107 | Qwen3.8-Flash-Next 176B BF16 TP8 | vllm | active |
| 8107 | Qwen3.8-Flash-Next 176B Q4_K_XL | llama.cpp | active (mutually exclusive with vLLM arm) |
| 8108 | GLM-5.3-Flash 320B Q8_0 | llama.cpp | active |
| 8109 | 176B BF16 splitKV | vllm | experimental |
| 8110 | Ornith-1.5-397B Q8_0 | llama.cpp | active |
| 8111 | Ornith-1.5-35B-A3B BF16 TP4 | vllm 0.28 | active |
| 8112 | DeepSeek-V4.1-Flash 748B Q4_K_M (resident) | llama.cpp | active |
| 8113 | Qwen3.8-27B BF16 TP2 | vllm 0.28 | active |
| 8114 | Qwen3.8-27B INT8-W8A8 TP1 + dflash12 | vllm 0.28 | active |
| 8203 | Qwen2.5-1.5B smoke | vllm 0.28 | active |
| 8301/8302 | 27B Fable splitKV / dflash2 A/B | vllm 0.28 | experimental |

Selection: single-stream fastest → 8107 llama.cpp Q4_K_XL; OpenAI API + big
KV pool → 8107 vLLM BF16; only resident large model → 8112. Full rationale in
`$AI/docs/recipes/README.md` §5 and each arm recipe §10.5.

## Data files

- `data/arms.json` — one entry per serving arm: identity, launcher entry,
  settled args/env, primary + other metrics with dates, patches, conflicts.
- `data/knobs.json` — cross-arm switches with settled values and decision
  rules (QuickReduce, MTP/spec depth, AIS/hipFile, `--fit off --load-mode dio`,
  TP split-mode, GPU visibility, tool-call/reasoning parsers, fused-MoE tiles).
- `data/environments.json` — rebuild + self-check for each `envs/*` tree.
- `data/patches.json` — apply/verify/revert gates per patch asset.
- `data/ops.json` — weight download, start/stop/observe, unsloth-studio.
- `data/recipe_kb.json` — Hyperloom RecipeKB seeding index (canonical ids +
  KB root + a compact per-row digest); consume via `KNOWLEDGE_LOCAL_ROOT` when
  launching Hyperloom.
- `data/vendor_references.json` — vendor (vllm-project/recipes) recipes folded
  in as *reference*: the vendor support matrix, the base args/env, and the
  explicit split between flags that transfer to gfx90a and what the vendor
  gates to MI300X+. Reference ≠ validated here; never quote one as an MI250X
  result.

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

## Hyperloom RecipeKB tie-in

This workspace also holds a RecipeKB corpus seeded from these arms:
root `/home/qiba/ROCm.AI/hyperloom/kb` (7-level dirs, `recipe.json` rows, hardware
pinned to `mi250x`). Point Hyperloom at it before launching an optimization
session so warm-start reads the workstation history:

```bash
export KNOWLEDGE_LOCAL_ROOT=/home/qiba/ROCm.AI/hyperloom/kb
```

Regenerate after any recipe change: `PYTHONPATH=/home/qiba/ROCm.AI/hyperloom
python3 /home/qiba/ROCm.AI/scripts/seed_recipe_kb.py` (arm rows),
`python3 /home/qiba/ROCm.AI/scripts/seed_reference_recipes.py` (vendor
reference rows + the Ornith arm fold-in), `python3
/home/qiba/ROCm.AI/scripts/note_emulation_boot.py` (boot evidence) and
`python3 /home/qiba/ROCm.AI/scripts/note_probe_crosscheck.py` (the measured
TTFT/throughput + run outcome; also repairs rows whose `what_worked` /
`what_failed` / `lessons` an optimizer CLOSE wiped), and `python3
/home/qiba/ROCm.AI/scripts/note_agent_lane.py` (the agent-lane numbers, the
`TPOT ~= 45.4 + 3.97ms x ctx/1024` law and `best_config` replay keys -- reads the
result JSONs under `hyperloom/.tmp/single_stream_lab`, archived in
`hyperloom/reports/fp8-emulation-probe/agent-lab/`) — all idempotent — then
`python3 /home/qiba/ROCm.AI/scripts/verify_recipe_kb.py` and
`python3 /home/qiba/ROCm.AI/scripts/build_skill_data.py` to refresh
`data/*.json`.

## aiter JIT prebuilt reuse (never pay the ~50 min compile twice)

aiter's JIT does **not** reuse across envs: a fresh env compiles 72 CK instances for
`module_gemm_a8w8` (~50 min of CPU, no GPU needed) on first import. Reuse *is* natively
supported — the switch is `AITER_JIT_DIR` — and this host has a verified cache.

```bash
cd ~/.cache/aiter-gfx90a
python3 restore.py --list                                       # 426 MB cache, 2 variants
python3 restore.py --env <env> --variant full-72inst-production --apply
HIP_VISIBLE_DEVICES=<idle die> python3 verify_gemm.py            # required acceptance
```

- Mechanism (read from aiter 0.1.19 source): `compile_ops` does `get_module(md)` and only
  falls into `build_module(...)` when that raises `ModuleNotFoundError`; with `AITER_JIT_DIR`
  set, `get_module_custom_op` imports the module from that dir (it is put on `sys.path[0]`).
  The `.so` must be the **bare name** `<md>.so` — `JIT_EXTENSION_VERSIONER` is per-process
  memory state whose first `bump_version_if_changed` returns 0, so there is no `_v1` suffix.
  `_needs_arch_rebuild()` scans the `.so` for `amdhsa--gfx*` and passes it when gfx90a is
  present. **Trap:** `build_module`'s `MainFunc` starts with
  `os.remove(get_user_jit_dir()/<md>.so)` — entering the compile path deletes a working
  prebuilt. "Compile finished but no `.so`" is that delete plus a re-compile, not lost output.
- **Three gates, all required:** aiter version + torch version + ROCm version must match the
  archive, and the running arch must be in the `.so` markers. Measured here: three envs share
  aiter 0.1.19 (8316 sources byte-identical), torch `2.12.0+git6bbd260`, ROCm 7.2.4.
- Verified install (`vllm_master_rocm724`, 2026-09-21): `get_module` **0.300 s** (not 50 min),
  `[256,4096,4096]` int8 GEMM rel_err **6.369e-03**, `[1024,2048,2048]` **7.042e-03**
  (tol 2e-2). Evidence line to look for in any arm's log:
  `[aiter] import [module_gemm_a8w8] under …/aiter/jit/module_gemm_a8w8.so`.
- Reading the numbers: a `max_abs` of 0.5 is **one bf16 ULP at magnitude ~256**, not an error —
  judge on **relative** error. And `not found tuned config in a8w8_tuned_gemm.csv, will use
  default config` means correctness is still proven but **performance is not** the production
  config; top up the tuning table before timing anything.
- **Do not** assume same-named `.o` files are interchangeable across envs: measured
  **38/38 same-name `.o` hashes differ** between `wu1w-int8-028` and `vllm_0.28.0_rocm72`,
  and the cause is a build-target difference (`"gfx90a": 104` in `GFX_CU_NUM_MAP`), not the
  ROCm version (both are 7.2.4). Copy `.so`, treat `.o` as a fallback only.
- Full write-up: `hyperloom/reports/aiter-jit-prebuilt-reuse.md`.

## 起服前的门 + 单臂探针（2026-09-21 新增，可直接复用）

- `quark-int8/gpu_gate.sh`：`source` 后 `gate 8121`（放行返回 0）/ `wait_free 8121 7200`。
  判据两条：除自己端口外无别的 `api_server` + 八张 GCD 各 <5 GiB（与 launcher 硬门同阈值）。
  内含三条单测与两个真实踩过的坑：`ps -eo args | grep -F '…api_server'` 会匹配 **grep 自己**
  ⇒ 门永远不过；`[ -ge $$… ]` 里的 `$$` 是 PID ⇒ 超时判断失效。
- `quark-int8/stack_probe.sh <臂名> KEY=VALUE …`：等卡 → 起服 → 事实召回 + TPS(conc 1/8/32)
  → 停服 → 追加 `logs/stack/results.jsonl`。fail-fast：每轮查容器 `State`，**连续两次**判死才撤，
  且**先把整份容器日志存盘再删容器**；`SKIP_BOOT=1` 可脱离 GPU 自测判据；
  `TPS_ISL_MULT=8/24` 切长上下文档（DCP/split-K 这类杠杆必须在长档判）。
- **一臂一杠杆**：混测只能探天花板，不能记收益。今天的教训：首轮五杠杆混测得 conc32 +47%，
  记在 split-K 头上；消融后真身是 `DSV41_IDX_AITER_KERNEL=1`（+69.3%），split-K 实际 −13%。
- 结论与数字出处：`hyperloom/reports/models/glm53-int4/decode-config-ablation.md`
  （机器可读同目录 `stack_results_20260921.jsonl`；`quark-int8/logs/` 是 gitignore 的）。
- 三个默认值即地雷，改 launcher 时别踩：① 枚举型 env 用 `${VAR:-}` 注入空串 ⇒ vLLM 抛
  `Invalid value ''`；② `MAX_CUDAGRAPH_CAPTURE_SIZE=0` + `ENFORCE_EAGER=0` ⇒ 断言拒绝（eager=0
  这条路此前从未走通）；③ `DSV41_IDX_AITER_KERNEL` 未设时走的是**本机不可信**的 torch 回退，
  而设 `=1` 实测 +69%。


## 与 serving-llms-on-instinct（AMD 官方技能）的仲裁（2026-09-21）

该技能直接读 `data/gpu_overrides.json > gpu_configs`（按 **gfx 架构**分档），而它只覆盖
`gfx942`/`gfx950`（MI300X/325X/350X/355X）—— 全技能内 MI250/gfx90a 命中 **0** 次。
所以落在本机的任务，它给的是通用/MI300 口径。**本机任务一律以本技能为准**，冲突处按下表：

| 通用口径可能怎么说 | 本机实测事实 |
|---|---|
| 启用 AITER 加速 | **不可用**（gfx90a 无 AITER MoE 路径） |
| 打开 QuickReduce | **禁用**：`init_custom_qr` 固定吃 ~9 GiB/卡 ≈ 本模型 KV 全部预算 |
| 用 split-KV 提速注意力 | **默认 0**：conc 1/8/32 三档全负（−18%/−14%/−13%） |
| 关掉 eager 换 CUDA graph | 需同时给 `MAX_CUDAGRAPH_CAPTURE_SIZE>=1`，否则断言拒绝 |
| indexer 走框架默认 | **必须 `DSV41_IDX_AITER_KERNEL=1`**：默认是更慢且本机不可信的回退，开启 +69.3%（conc32） |

- 桥接（幂等，bootstrap 还原第三方文件后要重跑）：
  `python3 hyperloom/patches-local/apply_serving_skill_mi250x.py`（`--check` 只看状态，`--revert` 回滚）。
- **SKU 级而非架构级**：64 GiB/GCD、104 CU/GCD、TP8 时权重 52.9 GiB/rank ⇒ 32k 档 KV 仅
  8.17 GiB / 94,016 tokens。别按「MI250X = 128 GiB」算 KV（那是两个 GCD 之和）。


## 本机运维增补（2026-09-21 第二轮：DCP/fp8KV、补丁队列、镜像与脚本自伤）

### DCP 与长上下文怎么起（launcher 没有 DCP 开关，走直通口）
- `VLLM_EXTRA_ARGS="--decode-context-parallel-size 8 --dcp-comm-backend ag_rs"`（上游只验证过 `ag_rs`）。
- 叠 fp8 KV：`--kv-cache-dtype fp8_e4m3`（只改 KV **存储**、无需 FP8 矩阵核；本机配置层已实测接受）。
- **KV 算术（本机唯一该用的口径）**：DCP=1 → 93.4 KiB/token（8.17 GiB = 94,016 tokens）；
  DCP=8 → **11.5 KiB/token**（5.89 GiB = 534,784 tokens，但权重 +2.11 GiB/rank）。
  1M 单序列需 1,048,576 tokens ⇒ **必须 fp8 KV × DCP=8 才过线**（bf16 KV 只有 534,784，差一半）。
- DCP=8 在短上下文（ISL≈700）是 **−28…−30%**，**必须在长档判**：`TPS_ISL_MULT=24`（≈17k）。

### 补丁队列（quark-int8/dcp_patches）自洽检查
- `python3 quark-int8/verify_patches.py`：①base+队列 与线上树**逐字节**相等 ②GEMV 四副本哈希
  ③split-K 默认 0 ④QR 赋值行未注释。
- 坑：`patch -i <相对路径>` 的路径按 **cwd** 解析 ⇒ 报"找不到补丁文件"、全 rc=2（曾被误归因为
  "多文件 hunk"）。必须传 `resolve()` 后的绝对路径。
- 树上还有 5 个文件被改过而 `base/` **无原件**（`weight_utils.py` + `models/deepseek_v41` 三个 +
  一个）⇒ 本门对它们**无判别力**，属已知盲区，别当成"已覆盖"。
- 新改动一律**追加**（如 0007/0008/0009），**不要重生成旧片**：重生成会让"队列==树"构造性为真，
  自检从此失去判别力，还会把后补的修复静默并进旧片、改写归属。生成+自证脚本：
  `quark-int8/dcp_patches/make_patch789.py`（生成后必须自证 sha256 相等）。

### Docker 自建镜像的坑（QR 镜像就是这么做的，也是这么踩的）
- `docker commit` 会把**被 commit 容器的 Entrypoint/Cmd 一并固化**。用 `--entrypoint bash` 起的
  bake 容器 commit 出来入口就是 `bash -c sleep …`，launcher 传的参数会被当脚本执行 —— 表现为
  容器**秒死**、日志只有一行 `/models: Is a directory`。正确做法显式还原并在事后逐行核对：
  `docker commit --change 'ENTRYPOINT ["vllm","serve"]' --change 'CMD ["bash"]' --change 'WORKDIR /app' <ctr> <img>`
  `docker inspect -f '{{.Config.Entrypoint}} {{.Config.Cmd}} {{.Config.WorkingDir}}' <img>`

### 数字口径（本机最容易记错账的地方）
- **请求级吞吐 ≠ 纯 decode tok/s**：`qr_tps_probe.py` 量的是含 prefill 的请求级（conc32 纪录 123.78），
  旧报告的 9.60–10.71 是纯 decode 口径，**两者不可互比**；只可同一把尺子内横比。
- 报告单流 TPS **必须带 ctx**：ctx≈800 约 10 tok/s，ctx=8192 约 4 tok/s。
- **内核倍数 ≠ 端到端**：split-K 内核 7.2×，端到端 −13%。

### 脚本自伤清单（今天为此烧掉两个 GPU 窗口，逐条都真发生过）
- `pkill -f <模式>` 会匹配**执行它的 shell 自己**（把自己杀掉）；`ps -eo args | grep -F '…api_server'`
  会匹配 **grep 自己** ⇒ 门永远不过。用 pgrep + 方括号模式 `api[_]server` 自保。
- `[ "$x" -ge $${VAR:-1} ]` 里的 `$$` 是 **PID** ⇒ 整数比较报错、超时判断整体失效。
- `bash -c` 里引用父 shell 变量必须 **export**，否则静默变空串（结果表写进空记录就是这来的）。
- 判死/删容器**之前先存日志**，否则现场消失；就绪判定必须**同时看容器 State**，否则会为秒死的
  容器空等满超时（曾空等 45 分钟，fail-fast 后单次失败反馈约 100 秒）。
## 本会话新增（2026-09-21 第三轮：MoE GEMV kernel、稀疏 split-K 开关、roofline 口径、KB 回填）

来源：GLM-5.3-CT-Int4-W4A16 / TP8+DCP8 / gfx90a 的实测与取证，报告在
`hyperloom/reports/models/glm53-int4/`。以下四条此前在本技能里 **0 命中**。

### MoE 专家 GEMV（自研 kernel，目前唯一已收回的 kernel 级杠杆）
- 问题：decode 是 M=1 的 GEMV，专家权重每个 token 全读一遍；把 **scale 提到 k 循环外**是
  这一步的关键（v3）。
- 开关（launcher 已接好，`PYTHONPATH=/patches/moe_gemv` 由 launcher 注入）：
  `MI250_MOE_GEMV=1`（默认开）、`MI250_MOE_GEMV_MODULE=mi250_moe_gemv_gs`、
  `MI250_MOE_GEMV_KERNEL`（v3 = scale hoisted）、`MI250_MOE_GEMV_BOTH`、`MI250_MOE_GEMV_DEBUG`。
- 实测：gemm1 **8.7×** / gemm2 **5.0×**（生产分片形状）；端到端单流 6.38–6.81 → **9.60–10.71 tok/s**；
  并发 32 聚合 33.8 → **59.1 tok/s**；事实召回仍 6/6。
- **v2 是被证伪的那一版**（单流 6.8 / 聚合 33.8 ≈ 等于不开），只留作对照；别把 `_v2` 当可用模块。
- 三处副本必须一致（补丁树 / `quark-int8/moe_gemv_patch/` / 容器），门是 `verify_patches.py ②`：
  `gs=af079db138ab` / `v2=c96264af84c1` / `v3=07b0d78f40b0` / `sitecustomize=31e8573f5500`。
- ✂️ **这条杠杆已经收回**：decode 归属表里 MoE GEMV 只占 **1.8%** 的步时间，别再去这里找收益。
  报告：`moe-gemv-scale-hoist.md`。

### 稀疏注意力 split-K（`MI250_SPARSE_SPLITK`）——与 0.28 的 split-KV **不是一回事**
- 本技能别处的「split-KV」指 `VLLM_ROCM_SPLITKV_PA`（paged-attention 的 KV 切分）；这里是
  **gfx90a 稀疏注意力路径自己的 split-K**：一条 launch 把 `(query, split)` 当行
  （`_splitk_make_indptr` 造 `[M*S+1]` indptr、查询优先行序 `r = i*S + s` ⇒ 源 ragged_indices
  无需重排），再用 `_splitk_merge` 做 LSE 合并。补丁：`quark-int8/dcp_patches/0009_gfx90a_sparse_splitk.patch`
  （队列可重放：`base + 0001..0009 == 线上树`，逐文件 sha256 相等）。
- 开关：`MI250_SPARSE_SPLITK=8`（**默认 0=关**）、`MI250_SPARSE_SPLITK_MAXM=8`（只在 M≤该值生效）。
- 结果：内核 7.2×（M=1）/ 2.4×（M=4），与 S=1 逐位一致（bf16 1 ulp）；但**端到端单流 −12%**
  （NCCL 141→255 µs、elementwise/copy 调用 1332→3439/step）⇒ 默认必须保持 0。
- ⚠️ 静默错陷阱：低层 `_rocm_sparse_attn_prefill_ragged_triton` 是**返回** out（内部 `empty_like(q)`），
  不能读预分配缓冲 —— 否则 out 全 0 而 lse 正常，看起来「没崩」但结果全错。

### Roofline 口径：本机的慢**不是带宽**问题（别再按带宽解释）
- `T_mem(mi250x) @num_gpus=8 isl=osl=1024 conc=32 = 859.0 tok/s`（BW 13.1 TB/s）；同参数 mi300x
  = 2779.3 ⇒ **比值 0.309 是防「跑在 MI300X 口径下」的假通过判据**。
- 实测对照（同一把尺子）：单流 @ctx≈800 = 10.2 tok/s（**1.19%**）；conc8 41.4（4.82%）；
  conc32 59.5（**6.93%**）。
- 原因：decode 一步 186.7 ms 里 sparse-attn 67.2（36%）、NCCL 36.3（19.4%）、
  `triton_w4a16_gemm` 35.3（18.9%）、我们的 MoE GEMV 3.3（1.8%）；权重流量只有
  **18.6 GB/s/rank = 峰值的 1.1%** ⇒ 受限在**层内串行/启动延迟**。
- 正确表述：「每步的层内串行开销吃掉 93% 的访存预算」，而不是「带宽不够」。
  报告：`decode-step-attribution.md`（含 roofline 一节）。

### RecipeKB 回填（GLM-5.3 int4 / mi250x）与三个静默坑
- 入口：`python3 scripts/note_glm53_int4_kb.py`（`--dry-run` 可预演；用 Hyperloom 自己的
  `LocalRecipeStore.put_recipe`，别手写 recipe.json，否则 `history/vN` 与 `version` 脱节）。
- 三个**静默**失败（今天全踩过）：① `remaining_gaps` 条目必须是 **dict**（`description`/`metrics`），
  写字符串会被直接丢弃；② `kernel_optimizations` 是**定长 dataclass**，键名不对会得到一串**全零**条目；
  ③ `best_config.extra_envs` 会被 warm-replay **当环境变量注入** ⇒ 别往里写说明文字。
- 权限坑：Hyperloom 容器以 root 写的槽位是 `root:root 0600`，宿主读不到 ⇒ `local_store.search()` 抛
  `LocalRecipeStoreError`、`verify_recipe_kb.py` 失败；修法：容器内 `chown -R 1000:1000` + `chmod 644`。
- 两条 warm-start 语义：`_DEFAULT_WARM_REPLAY_MIN_CONFIDENCE = 0.7`（低于它只被读、不会复现其
  `best_config`）；recipe 的 `what_failed` 会被注入 explore 的 rejected 账本 ⇒ **负结论写进去等于
  省下一次重测**。
- 磁盘行数可以多于 `seed_manifest.json`（实测 18 vs 12）：本库有**三种写入者**（播种 / Hyperloom 自己 /
  会话回填），见 `hyperloom/kb/README.md`。

### 待办（别当成已解决）
- QR（QuickReduce）在 **GLM-5.3 int4 上仍未验证**：A 臂基线已测（QR 关：召回 6/6；
  ISL≈800/OSL128 请求级 conc 1/8/32 = 4.38 / 28.15 / 74.77 tok/s —— 与 1024/1024 档的 59.1
  **不同尺子，不可互比**）；B 臂（QR **真开**）**至今没出过任何结果**。
- 进展（2026-09-21 10:40）：入口 bug 已被重烤修好（`docker inspect` 与 stock 镜像 Config 逐字一致，
  实跑 `vllm serve --help` 正常）。**A2 臂**（同一个 `-qr` 镜像、三条 env 全不设）已测：召回 6/6、
  请求级 TPS 4.18 / 26.64 / 73.13；对照 **A 臂**（stock 镜像）4.38 / 28.15 / 74.77 ⇒ 差 **−2…−5%**，
  落在本机 cross-boot 漂移内 ⇒ **仅凭单次对拍不能说"代码在但关着是中性"**，只能说没有反向证据。
  两臂日志里 `tp:0`/`ep:0` 都选 `['PYNCCL']`。
- 现在要跑 B 臂：`SKIP_A=1 SKIP_A2=1 bash quark-int8/qr_ab_watch.sh`（镜像 `...-0918-qr` 已可用；
  脚本里的 `wait_free` 会等到八张卡都空才动手）。

