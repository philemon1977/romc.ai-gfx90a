## 0. MANDATE

- deliverable: a source patch and/or up to 6 ranked config variants addressing the gap below
- anchor: `inference:qwen3.8-27b-abliterated-w8a8-gdnint8:unknown_gpu:vllm:qwen3_5:qwen3_5forconditionalgeneration:0.28.0:bf16#throughput_below_target`

Run status (read-only context; do NOT re-state these as your own measurements):
- baseline: 444.9 tok/s/GPU
- current best: 444.9 tok/s/GPU
- KEEP threshold this cycle: 1.00%

Judged by: the Coordinator benches your proposals end-to-end against
the sealed baseline and decides KEEP/REVERT; the accuracy gate runs
alongside. You are not asked to prove the number.

## 2. HARDWARE CONTEXT

- gpu_type: (none)
- TP: 1

Workload:
- precision: bf16
- concurrency: 64
- ISL (input seq len): 1024
- OSL (output seq len): 1024
- max_model_len: 6144

## 2a. EXECUTION BUDGET (wall-clock)

- Hard wall-clock budget for this entire dispatch: **600s (~10 min)**.
- Dispatch started at: 2026-09-14T19:00:50.997793+00:00 (UTC).
- The Coordinator hard-kills your subprocess when this budget is exhausted — turns are NOT the stop signal. Scope your work to reach a deliverable conclusion inside the budget.
- Self-throttle: check elapsed wall-clock with Bash (``date -u +%s`` vs the start above), keep your ``specialist_done.partial.json`` checkpoint current, and write the final ``specialist_done.json`` before the budget runs out so your best work is never lost to a kill.

## 3. GAP STATEMENT

- gap_canonical_id: `inference:qwen3.8-27b-abliterated-w8a8-gdnint8:unknown_gpu:vllm:qwen3_5:qwen3_5forconditionalgeneration:0.28.0:bf16#throughput_below_target`
- layer: framework
- symptom: Decode step 137.8ms at batch 64, ~6x off weight-BW roofline => compute/LSU-issue bound. AITER fused paths (unified-attn, fused QK-norm+RoPE+KV, GDN decode fast path) exist in the live tree but are gated off on gfx90a.

Most recent evidence:
```json
{
  "known_gates": "1) _get_backend_priorities only appends ROCM_AITER_UNIFIED_ATTN when is_aiter_found_and_supported(), which hard-requires get_cdna_version()>=3 \u2014 gfx90a is CDNA2. 2) AITER fused GDN decode fast path (_forward_core_decode_aiter: fused conv1d-update + fused_rearrange_sigmoid_gated_delta_rule) is gated on self.gqa_interleaved_layout (Qwen3-Next only); Qwen3.5 uses non-interleaved qkvz. 3) fused_reshape_causal_conv1d_update_single_token may hardcode interleaved layout \u2014 verify before assuming.",
  "live_source_root": "/opt/envs/wu1w/lib/python3.12/site-packages/vllm (launcher /usr/local/bin/vllm-wu1w) \u2014 NOT /opt/envs/vllm; static_recon specialist already verified this. Target the LIVE tree for any patch.",
  "profile_trace": "runs/baseline/e420b1b3a97f4b7eb334a1df38a32578/benchmark_vllm_20260914_173834/server.log"
}
```

## 4. KB CONTEXT (optional, advisory)

Structured KB context is empty for this (model, hardware, domain), but the research scout collected source-backed priors this session. Treat these as your advisory prior (co-equal with RecipeKB priors; the Critic still gates the final answer):

Proven priors collected by the research scout. Treat as advisory hints to try earlier — each carries a source.
- Baseline decode step costs 137.8ms at batch 64. Weight-streaming floor for 26.46GiB INT8 on one MI250X GCD (~1.23-1.6TB/s) is ~18-23ms/step, so we are ~6x off the memory roofline => the step is compute/LSU-issue bound, not weight-BW bound. Any lever that cuts per-step bytes-per-token or instruction count is worth more than another weight-BW lever. (impact=Reframes the search: target per-step decode work, not KV/util tuning., accuracy_risk=none (analysis), source=runs/baseline/e420b1b3a97f4b7eb334a1df38a32578/benchmark_vllm_20260914_173834/summary.txt + gpu_worker.py:741 memory line)
- GDN recurrent (SSM) state cache is float32. text_config.mamba_ssm_dtype='float32' and vllm/model_executor/models/config.py:786-805 (Qwen3_5ForConditionalGenerationConfig) copies it into cache_config.mamba_ssm_cache_dtype when that is 'auto'. Size: 48 value heads x 128 x 128 x 4B = 3.1MB/layer x 48 GDN layers = 151MB per in-flight request, read+written every decode step => ~19.3GB/step at conc 64 (~12-16ms/step at BW floor, and fp32 doubles the LSU/vector-register pressure at fixed sclk). --mamba-ssm-cache-dtype is a first-class CLI flag (engine/arg_utils.py:1237, MambaDType=['auto','float32','float16','bfloat16']) and an explicit user override is honoured with only a warning. (impact=Highest-value single byte-reduction available; 5-15% throughput if BW-relevant, more if LSU-bound at locked clocks., accuracy_risk=Real: bf16 recurrent state accumulates rounding over long (2048-6144) contexts. Model author shipped fp32 deliberately., source=/mnt/stripe-3mix-3t2/models/davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8/config.json + /opt/envs/vllm/lib/python3.12/site-packages/vllm/model_executor/models/config.py:781-805)
- GPU clocks were reported at sclk 800 MHz / fclk 400 MHz for ALL 1748 monitoring samples across the 3766s baseline run (range '800 - 800'), at a flat 96-98W package power, and a read-only `rocm-smi --showclocks` at idle now still shows sclk level 1 (800MHz) on every GCD. MI250X boost is ~1.7GHz/300W+. If the DPM level is genuinely pinned to min, every compute/issue-bound kernel in this workload pays ~2.1x and no vLLM flag can recover it. (impact=Potentially larger than all config levers combined; needs Coordinator confirmation because it is a rig property, not a serving flag., accuracy_risk=none, source=summary.txt 'GPU Clock: 800 - 800 MHz (avg: 800)' + rocm-smi --showclocks (read-only))
- Prefill interference is NOT the bottleneck. At steady state ~70 decode steps + ~2 prefill chunks (max_num_batched_tokens=2048 default, config/scheduler.py:42) fit in a 10s window; solving 70*T_dec + 2*T_pre = 10000ms with T_dec=137ms leaves only ~410ms total for prefill => mixed/prefill steps are ~4% of wall clock. Mean ITL (137.81ms) == mean TPOT (137.81ms) confirms this. Levers aimed at prefill/decode mixing (chunked-prefill budgets, cudagraph_mode for prefill) are low-yield here. (impact=Saves ~2 benchmark slots by ruling out the prefill-mixing family., accuracy_risk=none, source=runs/baseline/.../server.log loggers.py:310 lines (Avg prompt ~1000-1400 tok/s vs Avg generation ~140-290 tok/s at Running: 64))
- KV cache is NOT the limiter either: 31.96 GiB KV / 292,717 tokens, usage peaked at 58.7%, zero 'preempt'/'retract' occurrences in 2500+ scheduler lines, Running: 64 sustained. '--kv-cache-memory=37216046080 to fully utilize gpu memory' and gpu-memory-utilization tuning will produce 0% because nothing is being evicted. (impact=Rules out the memory-utilization family., accuracy_risk=none, source=runs/baseline/.../server.log lines 70-72,244 + grep -c preempt=0)
- KB law 'MTP is not built into 0.28.0+rocm723 here' is WRONG for this checkpoint. The target checkpoint SHIPS MTP weights (mtp.fc.weight, mtp.layers.0.{q,k,v,o_proj,q_norm,k_norm,mlp.*}) in model.safetensors.index.json, text_config has mtp_num_hidden_layers=1, and vllm/model_executor/models/qwen3_5_mtp.py + a registry entry exist. So --speculative-config {"method":"mtp","num_speculative_tokens":N} should load with NO external draft path. (impact=Unlocks a spec-decode lane for future rounds on a prose corpus. NOT proposed this round: the sealed benchmark uses dataset_name='random' and KB law (8) says random corpora collapse acceptance, so it would benchmark as a false negative., accuracy_risk=none (finding), source=/mnt/.../model.safetensors.index.json + vllm/model_executor/models/qwen3_5_mtp.py)
- The sealed harness benchmark is vllm bench serve with dataset_name='random', random_input_len=1024, random_output_len=1024, random_range_ratio=1.0, max_concurrency=64, ignore_eos=True, num_prompts=320, num_warmups=128 (GEMM/graph-config-valid corpus per KB law (9); invalid for judging spec decode). Prefix cache hit rate reaches 32.8% purely from warmup/main prompt reuse, so --no-enable-prefix-caching has a real prefill cost. (impact=Explains why random-corpus A/B is valid for config levers and constrains the prefix-caching proposal., accuracy_risk=none, source=runs/baseline/.../benchmark_stdout.log line 329 (argparse Namespace dump))

Anchor proposals on these hints where they fit the gap (Section 3) and hardware (Section 2).

## 4a. ROOFLINE EVIDENCE

(none — no fresh roofline snapshot has been recorded yet. The Coordinator auto-enqueues `roofline` at the end of PRELUDE and again after every 10% watermark crossing; if you are seeing this, the snapshot is still in-flight.)

## 5. WARM-START RECIPE SUMMARY

**find-recipe result:**
```json
{"confidence":0.0,"hw":"unknown_gpu","recipe":{"architectures":["Qwen3_5ForConditionalGeneration"],"authority":"EXPERIENTIAL","best_config":{},"best_throughput":0.0,"canonical_id":"inference:qwen3.8-27b-abliterated-w8a8-gdnint8:unknown_gpu:vllm:qwen3_5:qwen3_5forconditionalgeneration:0.28.0:bf16","conc":64,"confidence":0.85,"created_at":"2026-09-14T13:49:03.270640+00:00","ep":1,"evidence_refs":[],"framework_name":"vllm","framework_version":"0.28.0","hardware":"unknown_gpu","image_digest":"rocm-ai/vllm:0.28.0-rocm7.2.4","isl":1024,"kernel_optimizations":[],"last_profiled":"","lessons":[],"max_model_len":6144,"model":"Qwen3.8-27B-ABLITERATED-W8A8-gdnint8","model_class":"dense","model_type":"qwen3_5","osl":1024,"pitfalls":[],"precision":"bf16","provenance":{"details":{"sid":"20260914T173139Z-3ae15c89"},"generated_at":"2026-09-14T17:31:40.147526+00:00","generator":"t0_anchor","source":"hyperloom-inference-optimizer"},"remaining_gaps":[],"rocm_version":"7.2.4","sessions":[],"stack_fingerprint":{"aiter_commit":"","rocm_version":"7.2.4","vllm_version":"0.28.0+rocm723"},"tp":1,"updated_at":"2026-09-14T17:31:40.147813+00:00","version":6,"what_failed":[],"what_worked":[]},"tier":"seed_only","workload":"Qwen3.8-27B-ABLITERATED-W8A8-gdnint8"}
```

## 5b. RELATED LESSONS (prior KEEPs on this model+hw)

(none)

## 5c. KNOWN PITFALLS (do NOT repeat — prior REVERTs)

(none)

## 6. PR MONITOR

(unavailable: pr_monitor disabled)

## 7. LOCAL SOURCE NAVIGATION HINT

Installed source roots (read-only):
- /sgl-workspace/aiter/
- /sgl-workspace/sglang/
- /sgl-workspace/vllm/
- /app/ATOM/atom/
- /app/xDiT/
- /opt/envs/vllm/lib/python3.12/site-packages/
- /opt/envs/vllm/lib/python3.12/site-packages/aiter/
- /opt/envs/vllm/lib/python3.12/site-packages/aiter_meta/
- /opt/envs/vllm/lib/python3.12/site-packages/vllm/
- /opt/envs/vllm/lib/python3.1/site-packages/aiter/
- /opt/envs/vllm/lib/python3.1/site-packages/aiter_meta/
- /opt/envs/vllm/lib/python3.1/site-packages/vllm/
- /opt/rocm/

These trees are read-only. Use Read / Grep / Glob to navigate. Do NOT attempt Edit / Write / git apply on these trees.

Use ``WebSearch`` to look up the latest upstream version of the local repo and compare the implementation you intend to modify against what is there now. Use ``WebFetch`` to read the relevant file or PR directly — before authoring a patch, confirm whether the upstream repo already contains the fix or optimization you are about to write.
