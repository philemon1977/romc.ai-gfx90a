# gfx90a (MI250X) long-context decode attention — findings, audit, and the q>1 patch

Everything here was measured on this box: 8× MI250X (16 GCDs, gfx90a/CDNA2, 104 CUs
and ~1.3 TB/s HBM per GCD), vLLM 0.28.0 at
`/home/qiba/ROCm.AI/hyperloom/patches/fp8-w8a8-emulation-gfx90a/vllm`, model
Ornith-1.5-397B-FP8 (`qwen3_5_moe`: 60 layers, 45 GDN `linear_attention` +
**15 `full_attention`**, `num_key_value_heads=2`, `head_dim=256`, TP8 ⇒ 1 KV head
and 4 query heads per rank).

## 1. The wall, and what it is not

Single-stream decode TPOT measured with the **official** InferenceX client
(`benchmarks/vllm_mi250x.sh`, client-only mode), no speculation:

| context | TPOT | excess over the 1k floor | per 1k tokens of context |
|---|---|---|---|
| 1k | 45.4 ms | — | — |
| 32k | 171.1 ms | 125.7 ms | 3.93 ms |
| 128k | 553.9 ms | 508.5 ms | 3.97 ms |
| 240k | 967.9 ms | 922.5 ms | 3.98 ms |

`TPOT ≈ 45.4 + 3.97 ms × (ctx/1024)` predicts the three long points to ≤1.6 ms.
At 128k the KV a token must read is ~2.0 GiB/die and the kernel spends 0.509 s on
it: **4 GiB/s ≈ 0.30% of HBM peak**. Attention FLOPs there are ~8 GFLOP/token
(<1% of the 8-die BF16 peak). So this is neither bandwidth- nor compute-limited.

The cause is structural, and it is in the Python, not the silicon
(file:line in the tree above):

```
platforms/rocm.py:618 get_attn_backend_cls -> ROCM_ATTN (AITER needs cdna>2, _aiter_ops.py:134)
v1/attention/backends/rocm_attn.py:446     -> chunked_prefill_paged_decode(...)
v1/attention/ops/chunked_prefill_paged_decode.py:381
      use_rocm_custom_paged_attention(head 256) = False   # platforms/rocm.py:403 allows 64/128
      (and :403-style gate also fails because block_size=528 is not a power of two)
:440  rocm_splitkv_pa.try_paged_decode(...) = False       # VLLM_ROCM_SPLITKV_PA defaults OFF
:491  kernel_paged_attention_2d[(num_seqs=1, num_kv_heads=1)]   # <-- ONE CTA, serial over ctx
```

One program per (sequence, KV head) at batch 1 = 1 of 104 CUs doing anything.

## 2. The fix is already in the wheel — it is off by default

`v1/attention/ops/rocm_splitkv_pa.py` is a MI250X split-KV (flash-decoding)
implementation, already wired at the `:440` slot above, gated behind
`VLLM_ROCM_SPLITKV_PA=1`. Enabling it, measured with the same official client, live
server, **no** speculation:

| context | serial | split-KV | decode tok/s | gain |
|---|---|---|---|---|
| 1k | 45.4 ms | 38.8 ms | 25.8 | ×1.17 |
| 128k | 553.9 ms | 42.7 ms | 23.4 | **×13.0** |
| 240k | 967.9 ms | 44.7 ms | 22.4 | **×21.7** |

Slope drops from 3.97 to ~0.025 ms per 1k tokens (160× flatter) and the module hits
206–315 GB/s in a kernel-level microbench. Evidence it engaged: server log lines
`MI250X split-KV paged-decode 已接管` (32 in one round) and
`/tmp/sk_*.json` `takeover/reject_by_reason`.

**Consequence for the use case**: 256k single stream goes from unusable (0.94 tok/s
projected) to ~22 tok/s. Capacity was never the blocker there — `--max-num-seqs 16`
alone raises the KV pool from 154,624 to 452,748 tokens (and 592,560 at
`--max-model-len 262144` without MTP; MTP costs ~38% of it: 366,692).

## 3. The q>1 patch (speculative decoding × split-KV)

The gate's first condition was `max_query_len != 1 → reject`, so an MTP step
(3 query rows) never used split-KV: the two best levers could not stack.
`splitkv-q3.patch` (226 lines, one file) replaces that with
`1 <= max_query_len <= VLLM_ROCM_SPLITKV_PA_MAX_Q` (**default 1 ⇒ byte-identical
behaviour when unset**, verified) and extends both kernels:

* tile M dim = `(draft-token, q-head)` pairs: `nqp=4 × Q=3 = 12` rows still live in
  the existing 16-row padding ⇒ **the extra two tokens cost no second KV pass**;
* per-row causal bound `row_lim = seq_len - Q + 1 + tok`, applied in the tail block
  only (the hot loop keeps zero column masks via `part_end_safe`); the tail
  condition became the runtime test `part_start + n_full*BLOCK_N < part_end` — with
  `% BLOCK_N` the Q-difference could be skipped entirely;
* scratch "head" axis and reduce grid widened by `Q`, output addressed at
  `(token_idx + tok)`.

Validation — `test_splitkv_q3.py`, oracle = the *unmodified* module called 3× at
`max_query_len=1` with `seq_len = ctx - Q + 1 + tok`:

```
ctx=4096   PASS max_abs=0.0000 | ctx=32768  PASS max_abs=0.0000 | ctx=131072 PASS max_abs=0.0000
```

Bit-identity proves the plumbing (row mapping, per-token causality, scratch axis,
reduce grid); the numerics themselves were already established at Q=1 (split-KV and
the serial kernel agree bit-for-bit against a gathered fp32 reference).

Mechanism gain (`bench_in_tree_splitkv.py` style, one Q=3 launch vs three Q=1):

| context | 1×Q=3 | 3×Q=1 | ratio |
|---|---|---|---|
| 32k | 348.9 µs/layer | 987.9 µs | ×2.83 |
| 128k | 662.7 µs/layer | 1934.6 µs | ×2.92 |
| 240k | 967.0 µs/layer | 2813.9 µs | ×2.91 |

## 4. Two traps worth recording (both cost me a wrong conclusion first)

* **A microbench must reproduce the server's memory contract.** With a
  `ceil(ctx/528)`-column block table the kernel raised `Memory access fault by GPU`
  at 30720/32768/65536 (but not 131072 — the tell that it was luck of what was
  mapped after the allocation). Giving it the table width a live server always
  provides (`max_model_len/528` = 497 columns) made every size pass. The read past
  the table is real but unreachable in serving; still worth a defensive clamp
  upstream. Never report a kernel bug from a harness that misrepresents the caller.
* **`pgrep -f <name>` must not gate a pipeline.** An unreaped zombie
  (`pid 9772 ZN [bgsweep.sh] <defunct>`, ppid 1) matched `swee[p].sh` and idled 8
  GPUs for 60 minutes. Gate on positive completion markers, filter `ps -o stat=`
  for `Z`, and put a `timeout` on every blocking call
  (`.tmp/watchdog/watch.sh` + `check.sh` implement the dead-man switch).

## 4b. Where speculation's long-context cost actually was (round N, 2026-09-16)

Six rounds of end-to-end A/B had narrowed it to "one O(ctx) term per speculative step,
independent of k" and nothing more: raising the scratch budget (K2/K3) changed nothing,
and `k=1` vs `k=2` differed by 3.8% (L1: 435.1 vs 451.6 ms/step at 128k, against 42.7 ms
for split-KV with no speculation). One profiled window named it.

Method: `--profiler-config.profiler torch --profiler-config.torch_profiler_dir <dir>`
plus `POST /start_profile` / `POST /stop_profile`; each worker writes
`<dir>/profiler_out_<rank>.txt` = `key_averages().table(sort_by=self_cuda_time_total)`.
(`rocprofv3 --attach` is unusable on this host: it prints `:: success` and writes nothing,
with or without `LD_PRELOAD=librocprofiler-register.so`.) The window is one 145,426-token
prompt sent twice with prefix caching on, so the profiled request is almost pure decode.

| kernel (145k ctx, 48 greedy tokens) | MTP k=2 window | no speculation window |
|---|---|---|
| `_fwd_kernel.kd` (Triton prefix-prefill, `prefix_prefill.py:107`) | **7.485 s = 78.9%, 288 calls × 26.0 ms** | 376.9 ms, 15 calls × 25.1 ms (the one-time prefill — legitimate) |
| `kernel_paged_attention_splitkv.kd` | 81.8 ms, 290 calls × 0.28 ms | 194.0 ms, 705 calls × 0.275 ms = 4.1 ms/step |

Both kernels run on **every** speculative step. The reason is branch order in
`v1/attention/ops/chunked_prefill_paged_decode.py`:

```python
if max_query_len > 1:                       # a spec verify step has q = k+1 >= 2
    context_attention_fwd(...)              # Triton: scans the WHOLE context, per layer
                                            # ... and then execution FALLS THROUGH
...
elif _rocm_splitkv_pa.try_paged_decode(...):
    return                                  # split-KV overwrites those same output rows
```

So the Triton pass is 100% redundant whenever split-KV takes over, and it costs
16 layers × 26.0 ms per step. Arithmetic closes on the measured number:
`16 × 26.0 + 17 × 0.28 + ~16 ≈ 437 ms/step` vs `451.6 ms` measured (3%).

**Fix, and it is a dispatch fix rather than a new kernel** —
`multirow-skip-triton-prefill.patch` (137 lines, 1 file) adds
`_splitkv_multirow_takeover()` and calls it *before* `context_attention_fwd`; it returns
`False` for every shape the decode path would not have handed to split-KV anyway (native
ROCm paged attention available for the shape, `causal=False`/cross attention, q outside
`MAX_Q`, non-uniform rows), so default behaviour is unchanged. The behaviour it changes is
provably redundant work: the unpatched path's final output already comes from split-KV.

Status: applied to the overlay (`py_compile` clean, original kept as `.py.orig`),
**not yet measured end-to-end** — round P (official client at 1k/32k/128k/240k + the same
profiled window + a greedy-output hash comparison) is staged in
`.tmp/single_stream_lab/run_I_splitkv.py P` and is waiting for the GPUs, which another
container held at the operator's request. Expected: 451.6 ms/step → 40-60 ms at 128k,
i.e. ~2.9 tok/step ≈ 50-70 tok/s against today's best 23.4 tok/s.

## 5. Files

| file | role |
|---|---|
| `splitkv-q3.patch` | the reviewable patch (applies clean: `patch -p1 --dry-run` in the vllm dir) |
| `multirow-skip-triton-prefill.patch` | dispatch fix for multi-row (q>1) batches: try split-KV before the redundant Triton prefix-prefill (section 4b); regenerate with `make_multirow_patch.py [--apply]` |
| `draft_splitkv_q3.py` / `rocm_splitkv_pa.py.orig` | patched module (installed over the tree) and its backup |
| `test_splitkv_q3.py` | oracle-vs-patch correctness (the harness upstream is missing: no `tests/`, and the `bench/pa_splitkv.py`, `pa256_golden.py` the module cites are not on disk) |
| `bench_in_tree_splitkv.py`, `probe_splitkv.py` | kernel-level correctness/timing; `probe_splitkv.py` runs one config per process for crash bisection |
| `flash_decode.py`, `test_flash_decode.py` | **my independent re-implementation, currently WRONG** (disagrees with both in-tree kernels ⇒ a GQA group→KV-head or `x`-packing error in it). Kept as a cross-check scaffold, not usable. |

## 6. Open

1. ~~Round K showed speculation regressing long-context decode and the stats blamed a 3×
   scratch growth hitting the default 32 MiB budget.~~ **Closed, and the attribution was
   wrong twice over**: K2/K3 raised the budgets and changed nothing (scratch rejects went
   to zero, split-KV kept taking over), and section 4b shows the cost was a redundant
   Triton prefix-prefill pass, not a fallback. Remaining work is round P's end-to-end
   verification of the dispatch patch.
2. Host hygiene, both currently NOT done and both measurable in round M: cpufreq
   governor is `schedutil` (measured spread 2944–3509 MHz across policies at one
   instant; the Hyperloom preflight warned about it and it was ignored) and no rank
   is NUMA-bound (`Cpus_allowed_list=0-47`, host pages ~86% on node0 while 4 of 8
   GCDs sit on node1). Topology: node0 = PCI 11/14/31/34, node1 = 8e/93/ae/b3.
   Per-rank pinning is deliberately untested until a trustworthy rank→die map exists.
3. 1M context still needs weights kept 8-bit resident (tile-load dequant): the
   emulation kernel pins 48.27 GiB/die as BF16, leaving 6.9 GiB/die of KV against
   ~16 GiB needed.
