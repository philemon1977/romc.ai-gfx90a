# Preset: Ornith-1.5-397B-FP8 / 8x MI250X — agent ("vibe coding") lane

One-command launch for a Hyperloom optimization run aimed at **one interactive
agent stream with a long context**, with a hard envelope of **16 concurrent
sessions**. It is deliberately NOT the aggregate-throughput shape (`CONC=64`) the
stock CLI defaults to.

```bash
bash /home/qiba/ROCm.AI/hyperloom/presets/ornith-agent-longctx/launch.sh 5 30 --dry-run
bash /home/qiba/ROCm.AI/hyperloom/presets/ornith-agent-longctx/launch.sh 5 30
```

Args are `[max_hours] [target_gain]`; it refuses to start unless all 8 dies are
free, and writes the exact in-container command to `logs/inner_<stamp>.sh` plus
the transcript to `logs/run_<stamp>.log`.

## Lane shape

| knob | value | why |
|---|---|---|
| `--tp` | 8 | forced: weights are 386 GiB, so <7 dies cannot hold them |
| `--conc` | 1 | the objective is per-stream decode speed; batching depth is not the win here |
| `--isl` / `--osl` | 32768 / 1024 | agent turn with a real repo context |
| `--max-model-len` | 262144 | the checkpoint's trained context; fits one stream today (see capacity) |
| `--server-args` | `--language-model-only --max-num-seqs 16 --max-num-batched-tokens 8192 --enable-prefix-caching` | text-only frees HBM; `max-num-seqs 16` is what turns the KV pool from 154,624 -> 452,748 tokens |
| `--extra-env VLLM_BIN` | `patches/fp8-w8a8-emulation-gfx90a/bin/vllm-emulation` | carries the FP8->BF16 emulation patch into the *benchmark server* |
| `INFERENCE_OPTIMIZER_ASSET_ROOT` | `./assetroot` | private copy of the Magpie configs with `timeout_seconds: 1800` |
| `KNOWLEDGE_LOCAL_ROOT` | `hyperloom/kb` | warm-start reads the workstation KB (this lane's row is `experimental`, not `dead`) |

## The two defects this preset exists to route around

**1. The enablement patch never reached the benchmark server.**
Session `20260915T161838Z-ba2694ea` failed its baseline because `vllm_mi250x.sh`
booted the *stock* env: `runs/baseline/.../benchmark_vllm_20260915_190524/server.log`
ends in `RuntimeError: torch._scaled_mm is only supported on CUDA devices with
compute capability >= 9.0 or 8.9, or ROCm MI300+` -> `Engine core initialization
failed` -> `baseline_tput=0.0`, and the session then reported
`stop_reason=enablement_stalled` having measured nothing.

The obvious fixes are all blocked by design: `PYTHONPATH` is in
`BLOCKED_UNTRUSTED_ENV_NAMES` (`hyperloom/common/env_safety.py`), and that
blocklist gates every channel that could carry it — KB `best_config.extra_envs`
(`_config_replay_args_envs`), per-variant envs, `--extra-env` pins, and
`--reference-script` exports. The supported channel is a provisioned
`StackRuntime` (`stack_actions.to_runtime_override()` -> `pythonpath_prefix` ->
`apply_runtime_override()`, consumed at `baseline.py:3555`), which requires the
framework agent to actually build and keep that runtime.

This preset uses the operator-side equivalent instead: `VLLM_BIN` is *not* a
blocked key, `vllm_mi250x.sh` honours `${VLLM_BIN:-vllm}`, and the wrapper keeps
the patched tree in a file the operator owns:

```
patches/fp8-w8a8-emulation-gfx90a/
  vllm/                                  # full patched tree, usable as a PYTHONPATH root
  bin/vllm-emulation                     # sets PYTHONPATH + dedicated cache, execs /opt/envs/vllm/bin/vllm
  fp8-w8a8-emulation-gfx90a.patch        # 398-line diff vs installed vLLM 0.28.0, 4 files
```

The diff was generated from the installed tree and **verified to apply clean**
(`patch -p1 --dry-run` on a pristine copy). It touches:
`model_executor/kernels/linear/scaled_mm/emulation.py` (the kernel),
`scaled_mm/__init__.py` + `kernels/linear/__init__.py` (selection), and
`compilation/caching.py` (AOT-graph key invalidation — without it, a stale FP8
`[K,N]` graph replays into `assert_size_stride` at `profile_run`).

**2. The budget gate could never admit a baseline.**
`grid_runner` asks `session_grid_bounds()` for `variant_expected_sec`, which is
`boot_cost_sec + benchmark_cost_sec` — both derived from a *measured* baseline.
With no baseline, it falls back to the declared catastrophic-hang backstop, and
`baseline_vllm.yaml` ships `timeout_seconds: 7200` (AgentX raises it to 7800):
`2 remaining round(s) of 7800s` = 4.33 h, which no 3 h budget can satisfy. So the
run skipped every remaining variant (`variant not_run`) and stalled. The preset's
private asset root lowers the cap to `1800`, sized to this workload
(CONC=1, ISL 32768/OSL 1024, `NUM_PROMPTS` = 10 -> ~12-15 min/pass + ~12 min boot),
which makes admission fund a baseline plus several candidates inside 5 h.

Verify both fixes landed in a real run:

```bash
grep -a "Selected FP8W8A8EmulationLinearKernel" <session>/runs/baseline/*/benchmark_vllm_*/server.log
grep -aE "VLLM_BIN|EXTRA_VLLM_ARGS" <session>/runs/baseline/*/benchmark_vllm_*/config.yaml
grep -a "grid_runner: .*cannot fit" <run log>   # want: absent
```

## Capacity: where 256k / 1M actually stand

Measured on this route (config A boot, `--max-num-seqs 16`, `--gpu-memory-utilization
0.95`): `Available KV cache memory: 6.9 GiB` -> `GPU KV cache size: 452,748
tokens`, i.e. **16.0 KiB/token/die**. That matches the architecture: 60 layers, of
which only **15 are `full_attention`** (`full_attention_interval: 4`) and 45 are
GDN `linear_attention` (constant recurrent state, no per-token KV); `num_key_value_heads
= 2`, `head_dim = 256`, BF16 KV, sharded over TP8.

At 16.0 KiB/token/die against 6.9 GiB/die of KV budget:

| target | KV needed / die | status |
|---|---|---|
| 1 stream x 256k (trained cap) | 4.0 GiB | **fits today**; 452,748-token pool >= 262,144 |
| 16 streams x 32k | 7.8 GiB | short by ~0.9 GiB/die -> `--gpu-memory-utilization 0.98`, or FP8 KV |
| 16 streams x 27k | 6.6 GiB | the current 16-session ceiling (~28,297 tokens/session) |
| 1 stream x 1M | 16.0 GiB | **does not fit** with BF16-resident weights |

The 1M gap is a *weight-footprint* problem, not a kernel-speed problem: the
emulation kernel dequantizes FP8 -> BF16 **at load**, so 48.27 GiB/die is pinned
in HBM. Keeping the weights 8-bit resident and dequantizing at tile load (the
route proposed originally) frees ~23.9 GiB/die, which pays for 1M-token KV with
headroom and is the single change that unlocks both 1M and 16 x 64k. Note also
that 1M exceeds the checkpoint's trained `max_position_embeddings = 262144`, so it
additionally needs a rope scaling override (`partial_rotary_factor: 0.25`,
`rope_theta: 1e7`, mrope interleaved) and an accuracy check — capacity is the
easy half of that requirement.

Ordered as agreed: tune first (A/B/C measurements below), capacity phase second.

## Host hygiene, measured and currently NOT done

- cpufreq governor is `schedutil` (driver `acpi-cpufreq`, boost on); at one instant
  the policies sat at 2944 / 3493 / 3499 / 3509 MHz -- a ~19% core-to-core spread,
  i.e. exactly the jitter that contaminates p99. The Hyperloom preflight printed
  this warning during the last session and it was ignored. Round M re-measures the
  same point set with `performance` on.
- No NUMA binding: live workers show `Cpus_allowed_list=0-47`, host pages ~86% on
  node0 while 4 of 8 GCDs belong to node1. Topology: NPS1, 2 nodes;
  node0 = PCI 11/14/31/34, node1 = 8e/93/ae/b3 (2 whole cards each, no card
  straddles sockets). Per-rank pinning is deliberately **untested** until a
  trustworthy rank->die map exists -- guessing it would produce a wrong answer.
- Neither can touch the long-context wall (that was GPU-HBM-side and is now fixed);
  they affect the ~21-39 ms per-step floor and the tail percentiles.

## Measured

All numbers come from the **official** client (`benchmarks/vllm_mi250x.sh`,
client-only mode against a self-booted patched server): random dataset,
`--ignore-eos`, `--request-rate inf`, `--num-warmups 2*CONC`, `RANDOM_RANGE_RATIO=1`,
TP8, `--language-model-only --max-num-seqs 16 --max-num-batched-tokens 8192
--enable-prefix-caching`, `--gpu-memory-utilization 0.95`.

### Single-stream decode vs context  (CONC=1)

| context | no spec tok/s (TPOT) | MTP k=2 tok/s (TPOT) | speedup | TTFT (no spec) |
|---|---|---|---|---|
| 1k | 21.6 (45.4 ms) | 45.2 (21.2 ms) | **2.14x** | 1.13 s |
| 32k | 5.4 (171.1 ms) | 17.9 (55.9 ms) | **3.06x** | 11.05 s |
| 128k | 1.81 (553.9 ms) | 4.79 (208.7 ms) | **2.65x** | 23.88 s |
| 240k | 1.03 (967.9 ms) | 3.51 (284.9 ms) | **3.40x** | 68.27 s |

The collapse is a **law, not noise**: `TPOT ~= 45.4 ms + 3.97 ms x (ctx/1024)`
reproduces 32k/128k/240k within **1.6 ms** (predicted 172.6 / 554.2 / 967.5 vs
measured 171.1 / 553.9 / 967.9), and extrapolates to `262,144 -> 1063 ms =
0.94 tok/s`. But at 128k a token only has to read ~2.0 GiB/die of KV and the
kernel takes 0.509 s for it: **~4 GiB/s = 0.30% of HBM peak**. That is a
`ROCM_ATTN` paged-attention signature (no split-KV / flash-decoding), not a
hardware limit -- and it, not capacity, is what blocks 256k here.

### MTP acceptance depends on content (quote it with the workload)

| content | acceptance rate | mean acceptance length | measured speedup |
|---|---|---|---|
| official synthetic (`random` + `--ignore-eos` + greedy) | 88.4% | 2.72 (27% of windows saturated at 3.00) | 2.14-3.40x |
| real repo review turns (`/v1/chat/completions`, temp 0, natural stop) | **75.3%** | **2.51** (per-position 0.850/0.657) | **1.88x** (19.05 -> 35.86 tok/s) |

So the win transfers, but the synthetic number is ~13% optimistic on rate and
should never be quoted as the agent-lane answer.

### Concurrency envelope

CONC=16, ISL 1k: aggregate 138.1 tok/s (no spec) vs 133.6 tok/s (MTP) =>
**~8.6 tok/s per stream and MTP buys nothing at 16 streams**. The lane therefore
wants speculation at CONC=1 and none at the envelope -- which is exactly what
`num_speculative_tokens_per_batch_size` expresses.

### Capacity (16.0 KiB/token/die, measured: 6.9 GiB -> 452,748 tokens)

| shape | KV / die | status |
|---|---|---|
| 1 x 256k | 4.0 GiB | fits at `--max-num-seqs 16`; but see the 0.94 tok/s law above |
| 16 x 27k | 6.6 GiB | the current envelope ceiling |
| 16 x 32k | 7.8 GiB | short ~0.9 GiB/die -> `--gpu-memory-utilization 0.98` or FP8 KV |
| 1 x 1M | 16.0 GiB | needs weights kept 8-bit resident (frees ~23.9 GiB/die) + rope scaling beyond the trained 262,144 |

`--max-model-len` itself costs capacity: 262144 lowers the pool 452,748 ->
366,692 tokens (`1.40x` per 262k request), and vLLM hard-refuses `max_model_len >
max_position_embeddings` without `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.

### Prefix caching on agent turns

Shared 36k-token context, 4 turns: cold TTFT 13.17 s -> turn2 1.07 s -> turns 3-4
~3.6 s (median speedup 3.69x; 8.87x on the MTP server). Real-content probe:
first request 9.37 s -> later requests 2.0-3.1 s. Reuse decays because the GDN
hybrid cache only reuses page-aligned prefixes -- candidate fix to test:
`--mamba-cache-mode all`.

### What each round settled

| round | config | verdict |
|---|---|---|
| A | max-len 65536, no spec | single-stream baseline: 21.6 tok/s @1k |
| B | A + MTP k=2 | 2.00-2.61x on synthetic; nothing at conc 16 |
| C3 | max-len 262144, no spec | 128k = 1.55 tok/s; the 256k point failed on client overshoot |
| **D** | A minus prefix caching | **refuted the align-mode hypothesis** (5.25 vs A's 5.48 tok/s) -> the penalty is the attention kernel |
| B2 | C3 + MTP | 128k = 3.27 tok/s (2.65x) |
| F2 | ISL 237568 within 262144, no spec | 240k = 0.67 tok/s; real-content baseline 19.05 tok/s |
| G | F2 + MTP | 240k = 1.18 tok/s (3.40x); real-content 35.86 tok/s |

Reproduce: `.tmp/single_stream_lab/run_*.sh` (stages), `summarize.py` (table),
`realcode_probe.py` (content-sensitive acceptance), and the KB row's
`agent_lane` block, written by `scripts/note_agent_lane.py`.

### Not yet measured (next optimizer run's first candidates)

0. **DONE instead: the 3.97 ms/1k slope was a default-off kernel, not missing
   work.** `VLLM_ROCM_SPLITKV_PA=1` enables the MI250X split-KV (flash-decoding)
   decode kernel that already ships in this wheel and is dispatched one line
   before the serial 1-CTA fallback. Live, single stream, no speculation:

   | context | serial | split-KV | decode tok/s | gain |
   |---|---|---|---|---|
   | 1k | 45.4 ms | 38.8 ms | 25.8 | x1.17 |
   | 128k | 553.9 ms | 42.7 ms | 23.4 | **x13.0** |
   | 240k | 967.9 ms | 44.7 ms | 22.4 | **x21.7** |

   `q>1` extension (so it can stack with MTP) lives with its tests and audit in
   `hyperloom/kernels/gfx90a_flash_decode/`. Round K showed stacking regressing
   (161.2 ms @128k) because the 3x scratch hits the default 32 MiB budget and
   those shapes fall back to the serial kernel; K2 re-tests with
   `MAX_SCRATCH_MIB=192`. `VLLM_ATTENTION_BACKEND=TRITON_ATTN` was measured and is
   ~21% *slower* at 128k, so it is out.
2. MTP with k>2 and adaptive `num_speculative_tokens_per_batch_size`.
3. `--mamba-cache-mode all` for progressive prefix reuse.
4. Weights kept 8-bit resident with tile-load dequant (capacity phase).
5. No optimizer sealed baseline exists yet (`baseline_tput=0.0`), so
   `best_throughput` in the KB row stays 0.0 by design.
