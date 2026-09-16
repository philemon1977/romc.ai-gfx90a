## 0. MANDATE

- deliverable: findings and up to 6 ranked config variants (read-only; no patch)
- anchor: `gap.research_scout.round0`

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
- Dispatch started at: 2026-09-14T18:41:22.664194+00:00 (UTC).
- The Coordinator hard-kills your subprocess when this budget is exhausted — turns are NOT the stop signal. Scope your work to reach a deliverable conclusion inside the budget.
- Self-throttle: check elapsed wall-clock with Bash (``date -u +%s`` vs the start above), keep your ``specialist_done.partial.json`` checkpoint current, and write the final ``specialist_done.json`` before the budget runs out so your best work is never lost to a kill.

## 3. GAP STATEMENT

- gap_canonical_id: `gap.research_scout.round0`
- layer: research
- symptom: Collect proven priors (reference launch scripts, model config.json architecture features, cross-framework / NVIDIA research) into prioritised research hints with sources; do not benchmark or patch.

## 4. KB CONTEXT (optional, advisory)

(none)

(No structured KB context supplied. Use Sections 1, 3, 5, and 6 plus source inspection; record missing RecipeKB / research / PR questions in ``residual_questions`` so a future round can warm richer advisory context.)

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
