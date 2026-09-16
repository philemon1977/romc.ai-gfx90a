# Research Hints

## 1. Baseline 512.5 tok/s decomposes as ~729 tok/s steady-state decode (all-zero-prompt-throughput windows) vs ~27-278 tok/s generation during prefill windows: prefill and decode are almost fully serialized. Accounting: 5 waves x (38.5s prefill + 94.5s decode) = 665s vs the measured 639.3s duration. ~29% of wall clock buys zero output tokens, so prefill-side and decode-overlap levers are worth roughly the same as decode-speed levers on this workload.
- expected_impact: frames the whole remaining headroom: 512.5 -> 729 tok/s is +42%
- accuracy_risk: none (analysis only)
- domain_tags: roofline, scheduling, profiling
- status: proposed
- source: runs/baseline/51ac28ff8aed4ad59746fdb889fae7bb/benchmark_vllm_20260915_103510/{server.log,inferencex_result.json}

## 2. Capacity is NOT the binding constraint: GPU KV cache usage peaks at 57.3% and 'Waiting: 0 reqs' for most of the run, and the engine reports Prefix cache hit rate 38-76%. num_gpu_blocks is therefore over-provisioned for conc=64, so gpu-memory-utilization and KV-capacity tuning are dead ends on this workload - do not spend proposal slots on them.
- expected_impact: negative knowledge: frees slots
- accuracy_risk: none
- domain_tags: kv_cache, anti-pattern
- status: proposed
- source: runs/baseline/.../server.log loggers.py:310 lines 10:37:34-10:43:54

## 3. GDN recurrent state is fp32 because the checkpoint declares mamba_ssm_dtype='float32' and vLLM auto-adopts it, not because a kernel requires fp32. The state cache is the single largest per-step HBM traffic source at conc=64 (~75MB per request per decode step over 48 linear-attn layers) and the 784-token attention block size is inflated specifically to match that mamba page size. --mamba-ssm-cache-dtype is a first-class CLI flag in 0.28.0+rocm723 and accepts bfloat16/float16.
- expected_impact: 3-10% output tok/s; unblocks further page-size gains
- accuracy_risk: moderate: deviates from the vendor-declared fp32 state precision on a W8A8 checkpoint whose accuracy headroom is already thin (baseline 0.9674). The delta-rule recurrence may still accumulate in fp32 and only narrow the store, in which case the loss is sub-1-ulp per step, but it has not been measured here - recommend the accuracy gate run at a tightened floor before KEEP.
- domain_tags: memory_bandwidth, gdn, mamba, quantization
- status: proposed
- source: /opt/envs/vllm/lib/python3.12/site-packages/vllm/model_executor/models/config.py:781-805 + config/cache.py:38,133 + engine/arg_utils.py:1237

## 4. The decoder attention backend was selected by elimination, not merit: ROCM_ATTN was used only because TURBOQUANT was rejected for AttentionType.DECODER, and ROCM_ATTN itself logged a fallback to the Triton chunked_prefill_paged_decode kernel. TRITON_ATTN is the other candidate the selector deemed compatible, and head_dim=256 is within ROCM_ATTN's supported head sizes [32,64,80,96,128,160,192,224,256]. This is distinct from the AITER fmha_v3 ASM path that platform law 11 rules out for hd256.
- expected_impact: 3-10% via prefill speed
- accuracy_risk: none (same math, different kernel)
- domain_tags: attention, backend_selection, prefill
- status: proposed
- source: runs/baseline/.../server.log rocm.py:703 + chunked_prefill_paged_decode.py:463 + v1/attention/backends/rocm_attn.py:193

## 5. MTP is NOT a dead end for framework support but IS unjudgeable on this benchmark: the checkpoint ships 15 usable 'mtp.*' weights (bf16, listed in quantization_config.ignore), 'qwen3_5_mtp' is a registered MTPModelTypes value, and speculative.py:551-572 auto-derives n_predict from text_config.mtp_num_hidden_layers=1 with architecture Qwen3_5MTP. This directly contradicts the model_arch.json note 'mtp: not built into 0.28.0+rocm723 here' - the model file qwen3_5_mtp.py exists and is in the registry. The reason it is NOT in proposal_set is platform law 8: the sealed corpus is random-token (RANDOM_RANGE_RATIO=1), which collapses draft acceptance, so a measured number here cannot credit or discredit it.
- expected_impact: potentially large on natural prose, unmeasurable here
- accuracy_risk: none if acceptance-correct sampling; lossless by construction
- domain_tags: speculative_decoding, mtp, corrected-kb-fact
- status: proposed
- source: /opt/envs/vllm/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5_mtp.py + config/speculative.py:50,551-572 + model.safetensors.index.json mtp.* keys

## 6. Environment/tooling gaps for this round: INFERENCEX_PATH is unset and state.json active_inferencex_path is empty, so reference launch scripts (source 1) were unreachable; pr_monitor is disabled (pr_degraded_reason ir3_auto), so source 3 upstream PR survey was unavailable. Findings are therefore 100% from model config.json + installed-source inspection + baseline run artifacts. Note also AITER_LOG_TUNED_CONFIG=1 is set in the base env but is pure log noise given law 5.
- expected_impact: process
- accuracy_risk: none
- domain_tags: process, kb-gap
- status: proposed
- source: manifest.json dependencies.inferencex.path + state.json active_inferencex_path + Section 6

## 7. Warm-start recipe is seed_only with confidence 0.0 and hardware 'unknown_gpu', best_throughput 0.0, and empty lessons/pitfalls/what_worked - no prior validated config exists for this canonical_id, so nothing here was derived from a RecipeKB win. Real hardware is gfx90a (manifest.json gfx_arch), i.e. the recipe's hardware field is mislabelled and should be corrected on writeback.
- expected_impact: process
- accuracy_risk: none
- domain_tags: recipekb, metadata
- status: proposed
- source: Section 5 find-recipe JSON + manifest.json gfx_arch=gfx90a

## 8. Base launch args leave the multimodal stack loaded in a wasteful configuration: --language-model-only IS present (good), but VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 and CUDA_VISIBLE_DEVICES deprecation warnings ('support will be removed in vLLM v0.26.0. Please use HIP_VISIBLE_DEVICES instead', rocm.py:134, emitted twice from APIServer and EngineCore) are logged every launch. Cosmetic, but the HIP_VISIBLE_DEVICES migration is a real forward-compat item if the image is ever bumped.
- expected_impact: 0 now
- accuracy_risk: none
- domain_tags: hygiene
- status: proposed
- source: runs/baseline/.../server.log rocm.py:134

## 9. Three Triton kernels JIT-compiled DURING the measured inference window (jit_monitor.py:141: _fwd_kernel, batch_memcpy_kernel, fused_sigmoid_gating_delta_rule_update_kernel) and the log's own advice is 'consider extending warmup to cover this shape/config'. NUM_WARMUPS=8 with 320 prompts; the first measured window (10:37:34) shows 239.9 gen tok/s, consistent with cold-shape compilation being charged to the measurement. fused_sigmoid_gating_delta_rule_update_kernel is a GDN decode kernel, so its cold compile lands on the decode path itself.
- expected_impact: 1-3% via warmup coverage (benchmark-side, not server-side)
- accuracy_risk: none
- domain_tags: warmup, jit, gdn
- status: proposed
- source: runs/baseline/.../server.log 10:37:23 jit_monitor.py:141
