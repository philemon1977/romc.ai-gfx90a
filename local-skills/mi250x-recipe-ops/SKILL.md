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

