## 0. MANDATE

- deliverable: findings and up to 6 ranked config variants (read-only; no patch)
- anchor: `inference:qwen3.8-27b-abliterated-w8a8-gdnint8:unknown_gpu:vllm:qwen3_5:qwen3_5forconditionalgeneration:0.28.0:bf16#throughput_below_target`

Run status (read-only context; do NOT re-state these as your own measurements):
- baseline: 512.5 tok/s/GPU
- current best: 512.5 tok/s/GPU
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
- Dispatch started at: 2026-09-15T11:41:22.222844+00:00 (UTC).
- The Coordinator hard-kills your subprocess when this budget is exhausted — turns are NOT the stop signal. Scope your work to reach a deliverable conclusion inside the budget.
- Self-throttle: check elapsed wall-clock with Bash (``date -u +%s`` vs the start above), keep your ``specialist_done.partial.json`` checkpoint current, and write the final ``specialist_done.json`` before the budget runs out so your best work is never lost to a kill.

## 3. GAP STATEMENT

- gap_canonical_id: `inference:qwen3.8-27b-abliterated-w8a8-gdnint8:unknown_gpu:vllm:qwen3_5:qwen3_5forconditionalgeneration:0.28.0:bf16#throughput_below_target`
- layer: framework
- symptom: current_best is 30.0% short of the run objective target on a hybrid GDN (Qwen3.5-architecture) W8A8 model on vLLM 0.28.0+rocm723 / gfx90a.

Most recent evidence:
```json
{
  "deliverable": "ranked candidate list, each marked already-present-in-0.28.0 / not-applicable / benchable-as-config / needs-patch, with concrete evidence links",
  "note": "pr_monitor was disabled last round (ir3_auto) and INFERENCEX_PATH unset, so survey via WebSearch/WebFetch against upstream vllm-project/vllm instead. Local installed source is /opt/envs/vllm/lib/python3.12/site-packages/vllm/ (live tree per static_recon is /opt/envs/wu1w).",
  "targets": [
    "PRs/commits on mamba-ssm-cache-dtype / GDN recurrent state bf16 or fp16 support and whether the fused kernels (qwen_gdn_linear_attn.py fused_recurrent_gated_delta_rule_packed_decode, fused_sigmoid_gating_delta_rule_update) consume bf16 state directly or fall back (this is the residual question the bf16 state proposal depends on)",
    "PRs on attention backend selection for ROCm gfx90a head_dim=256 / TRITON_ATTN vs ROCM_ATTN performance",
    "PRs on chunked-prefill/decode overlap for hybrid mamba/GDN models (mamba_cache_mode=align scheduler behavior)",
    "PRs on VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE behavior on CDNA2",
    "whether the AITER CDNA3-only gate (is_aiter_found_and_supported, get_cdna_version()>2) has any upstream carve-out or env override that would enable GDN decode fast paths on gfx90a without a source patch"
  ]
}
```

## 4. KB CONTEXT (optional, advisory)

Structured KB context is empty for this (model, hardware, domain), but the research scout collected source-backed priors this session. Treat these as your advisory prior (co-equal with RecipeKB priors; the Critic still gates the final answer):

Proven priors collected by the research scout. Treat as advisory hints to try earlier — each carries a source.
- Baseline 512.5 tok/s decomposes as ~729 tok/s steady-state decode (all-zero-prompt-throughput windows) vs ~27-278 tok/s generation during prefill windows: prefill and decode are almost fully serialized. Accounting: 5 waves x (38.5s prefill + 94.5s decode) = 665s vs the measured 639.3s duration. ~29% of wall clock buys zero output tokens, so prefill-side and decode-overlap levers are worth roughly the same as decode-speed levers on this workload. (impact=frames the whole remaining headroom: 512.5 -> 729 tok/s is +42%, accuracy_risk=none (analysis only), source=runs/baseline/51ac28ff8aed4ad59746fdb889fae7bb/benchmark_vllm_20260915_103510/{server.log,inferencex_result.json})
- Capacity is NOT the binding constraint: GPU KV cache usage peaks at 57.3% and 'Waiting: 0 reqs' for most of the run, and the engine reports Prefix cache hit rate 38-76%. num_gpu_blocks is therefore over-provisioned for conc=64, so gpu-memory-utilization and KV-capacity tuning are dead ends on this workload - do not spend proposal slots on them. (impact=negative knowledge: frees slots, accuracy_risk=none, source=runs/baseline/.../server.log loggers.py:310 lines 10:37:34-10:43:54)
- GDN recurrent state is fp32 because the checkpoint declares mamba_ssm_dtype='float32' and vLLM auto-adopts it, not because a kernel requires fp32. The state cache is the single largest per-step HBM traffic source at conc=64 (~75MB per request per decode step over 48 linear-attn layers) and the 784-token attention block size is inflated specifically to match that mamba page size. --mamba-ssm-cache-dtype is a first-class CLI flag in 0.28.0+rocm723 and accepts bfloat16/float16. (impact=3-10% output tok/s; unblocks further page-size gains, accuracy_risk=moderate: deviates from the vendor-declared fp32 state precision on a W8A8 checkpoint whose accuracy headroom is already thin (baseline 0.9674). The delta-rule recurrence may still accumulate in fp32 and only narrow the store, in which case the loss is sub-1-ulp per step, but it has not been measured here - recommend the accuracy gate run at a tightened floor before KEEP., source=/opt/envs/vllm/lib/python3.12/site-packages/vllm/model_executor/models/config.py:781-805 + config/cache.py:38,133 + engine/arg_utils.py:1237)
- The decoder attention backend was selected by elimination, not merit: ROCM_ATTN was used only because TURBOQUANT was rejected for AttentionType.DECODER, and ROCM_ATTN itself logged a fallback to the Triton chunked_prefill_paged_decode kernel. TRITON_ATTN is the other candidate the selector deemed compatible, and head_dim=256 is within ROCM_ATTN's supported head sizes [32,64,80,96,128,160,192,224,256]. This is distinct from the AITER fmha_v3 ASM path that platform law 11 rules out for hd256. (impact=3-10% via prefill speed, accuracy_risk=none (same math, different kernel), source=runs/baseline/.../server.log rocm.py:703 + chunked_prefill_paged_decode.py:463 + v1/attention/backends/rocm_attn.py:193)
- MTP is NOT a dead end for framework support but IS unjudgeable on this benchmark: the checkpoint ships 15 usable 'mtp.*' weights (bf16, listed in quantization_config.ignore), 'qwen3_5_mtp' is a registered MTPModelTypes value, and speculative.py:551-572 auto-derives n_predict from text_config.mtp_num_hidden_layers=1 with architecture Qwen3_5MTP. This directly contradicts the model_arch.json note 'mtp: not built into 0.28.0+rocm723 here' - the model file qwen3_5_mtp.py exists and is in the registry. The reason it is NOT in proposal_set is platform law 8: the sealed corpus is random-token (RANDOM_RANGE_RATIO=1), which collapses draft acceptance, so a measured number here cannot credit or discredit it. (impact=potentially large on natural prose, unmeasurable here, accuracy_risk=none if acceptance-correct sampling; lossless by construction, source=/opt/envs/vllm/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5_mtp.py + config/speculative.py:50,551-572 + model.safetensors.index.json mtp.* keys)
- Environment/tooling gaps for this round: INFERENCEX_PATH is unset and state.json active_inferencex_path is empty, so reference launch scripts (source 1) were unreachable; pr_monitor is disabled (pr_degraded_reason ir3_auto), so source 3 upstream PR survey was unavailable. Findings are therefore 100% from model config.json + installed-source inspection + baseline run artifacts. Note also AITER_LOG_TUNED_CONFIG=1 is set in the base env but is pure log noise given law 5. (impact=process, accuracy_risk=none, source=manifest.json dependencies.inferencex.path + state.json active_inferencex_path + Section 6)
- Warm-start recipe is seed_only with confidence 0.0 and hardware 'unknown_gpu', best_throughput 0.0, and empty lessons/pitfalls/what_worked - no prior validated config exists for this canonical_id, so nothing here was derived from a RecipeKB win. Real hardware is gfx90a (manifest.json gfx_arch), i.e. the recipe's hardware field is mislabelled and should be corrected on writeback. (impact=process, accuracy_risk=none, source=Section 5 find-recipe JSON + manifest.json gfx_arch=gfx90a)
- Base launch args leave the multimodal stack loaded in a wasteful configuration: --language-model-only IS present (good), but VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 and CUDA_VISIBLE_DEVICES deprecation warnings ('support will be removed in vLLM v0.26.0. Please use HIP_VISIBLE_DEVICES instead', rocm.py:134, emitted twice from APIServer and EngineCore) are logged every launch. Cosmetic, but the HIP_VISIBLE_DEVICES migration is a real forward-compat item if the image is ever bumped. (impact=0 now, accuracy_risk=none, source=runs/baseline/.../server.log rocm.py:134)
... and 1 more in research_hints.md.

Anchor proposals on these hints where they fit the gap (Section 3) and hardware (Section 2).

## 4a. ROOFLINE EVIDENCE

(none — no fresh roofline snapshot has been recorded yet. The Coordinator auto-enqueues `roofline` at the end of PRELUDE and again after every 10% watermark crossing; if you are seeing this, the snapshot is still in-flight.)

## 5. WARM-START RECIPE SUMMARY

**find-recipe result:**
```json
{"confidence":0.0,"hw":"unknown_gpu","recipe":{"architectures":["Qwen3_5ForConditionalGeneration"],"authority":"EXPERIENTIAL","best_config":{},"best_throughput":0.0,"canonical_id":"inference:qwen3.8-27b-abliterated-w8a8-gdnint8:unknown_gpu:vllm:qwen3_5:qwen3_5forconditionalgeneration:0.28.0:bf16","conc":64,"confidence":0.85,"created_at":"2026-09-14T13:49:03.270640+00:00","ep":1,"evidence_refs":[],"framework_name":"vllm","framework_version":"0.28.0","hardware":"unknown_gpu","image_digest":"rocm-ai/vllm:0.28.0-rocm7.2.4","isl":1024,"kernel_optimizations":[],"last_profiled":"","lessons":[],"max_model_len":6144,"model":"Qwen3.8-27B-ABLITERATED-W8A8-gdnint8","model_class":"dense","model_type":"qwen3_5","osl":1024,"pitfalls":[],"precision":"bf16","provenance":{"details":{"sid":"20260915T103412Z-bf20aced"},"generated_at":"2026-09-15T10:34:13.113448+00:00","generator":"t0_anchor","source":"hyperloom-inference-optimizer"},"remaining_gaps":[],"rocm_version":"7.2.4","sessions":[],"stack_fingerprint":{"aiter_commit":"","rocm_version":"7.2.4","vllm_version":"0.28.0+rocm723"},"tp":1,"updated_at":"2026-09-15T10:34:13.113600+00:00","version":7,"what_failed":[],"what_worked":[]},"tier":"seed_only","workload":"Qwen3.8-27B-ABLITERATED-W8A8-gdnint8"}
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
