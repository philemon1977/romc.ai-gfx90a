# Inference Optimizer Report — Ornith-1.5-397B-FP8_20260915T161838Z_8e79dc43

- **Model**: Ornith-1.5-397B-FP8  (`/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-FP8`)
- **Stop reason**: `enablement_stalled`
- **Why it stopped**: The enablement loop made no forward progress for several consecutive rounds and stopped instead of re-deriving the same fix.
- **⚠ Degraded mode**: ran on the TEXT path only (multimodal inputs ignored) — see 'Degraded mode' below
- **Budget**: 180 minutes
- **Generated**: 2026-09-15T19:16:06.873069+00:00

## Throughput

- baseline            : `0.0 tok/s/GPU`
- cumulative_gain_val : `0.00%` ⚠ never validated — nothing has promoted in this session

## Run summary

- crash_count    : 0
- pruned_families: (none)
- host           : amd-server
- platform       : AMD EPYC 7413 24-Core Processor — SMT off, NPS1 (2 NUMA nodes / 2 sockets), governor schedutil, boost on
- accelerators   : 8× gfx90a on the host, amdgpu 6.16.13
- stack          : aiter unknown, rocm 7.2.4, sglang unknown, vllm 0.28.0+rocm723, kernel 6.8.0-139-generic

## Event counts

- `observation`: 38
- `alert`: 22
- `delegated_result`: 14
- `heartbeat`: 14
- `decision`: 8
- `event`: 8
- `proposal`: 8
- `review_verdict`: 8
- `advice`: 2
- `strategy_change`: 1

## Highlights

- `alert` from `robustness`: sev=medium agent coordinator hit policy_denied 4 times (>= 3); top rule=enablement_round_in_flight
- `alert` from `robustness`: sev=medium log error pattern 'RuntimeError' matched 8 times
- `alert` from `robustness`: sev=high log error pattern 'RuntimeError: Engine core initialization failed' matched 2 times
- `alert` from `robustness`: sev=high wall-clock budget 98% consumed (elapsed=176.5min / budget=180min) and cumulative_gain_validated=0.00% — wind the session down before the deadline cuts it
- `alert` from `robustness`: sev=medium PRELUDE phase budget 3243% consumed (10506s of 324s cap)
- `alert` from `robustness`: sev=high hard deadline cutoff: only 4.9min remain (<= 5min); finalize now
- `alert` from `robustness`: sev=medium aiter jit build_count=6 unchanged for 10 consecutive ticks; a prior hipcc invocation likely crashed mid-build
- `delegated_result` from `coordinator`: kind=baseline state=succeeded tput=None decision=None
- `decision` from `coordinator`: kind=approved_proposal action=baseline task=1d343fd9
- `review_verdict` from `critic`: verdict=approve reason=PRELUDE baseline is an exploration/framework operation whose
- `proposal` from `orchestration`: action_name=baseline
- `delegated_result` from `coordinator`: kind=integrate_patch state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=medium agent coordinator hit policy_denied 3 times (>= 3); top rule=enablement_round_in_flight
- `alert` from `robustness`: sev=medium log error pattern 'RuntimeError' matched 8 times
- `alert` from `robustness`: sev=high log error pattern 'RuntimeError: Engine core initialization failed' matched 2 times
- `alert` from `robustness`: sev=high only 15.2min remain (<= 30min); validated_gain still 0; wind the session down now
- `alert` from `robustness`: sev=high wall-clock budget 92% consumed (elapsed=164.8min / budget=180min) and cumulative_gain_validated=0.00% — wind the session down before the deadline cuts it
- `decision` from `coordinator`: kind=approved_proposal action=integrate_patch task=46182654
- `review_verdict` from `critic`: verdict=advise reason=This is classified as enablement_landing with specialist pro
- `delegated_result` from `coordinator`: kind=recover state=succeeded tput=None decision=None
- `proposal` from `coordinator`: action_name=integrate_patch
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=high local server probe http://127.0.0.1:8888/health status=error
- `delegated_result` from `coordinator`: kind=baseline state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=medium PRELUDE phase budget 1992% consumed (6455s of 324s cap)
- `decision` from `coordinator`: kind=approved_proposal action=baseline task=9d87461a
- `review_verdict` from `critic`: verdict=approve reason=Baseline is an expected PRELUDE framework_op/exploration act
- `proposal` from `orchestration`: action_name=baseline
- `delegated_result` from `coordinator`: kind=integrate_patch state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=medium wall-clock budget 59% consumed with cumulative_gain_validated=0.00% — early strategy hint: cheap actions are not paying off
- `alert` from `robustness`: sev=medium aiter jit build_count=6 unchanged for 5 consecutive ticks; a prior hipcc invocation likely crashed mid-build
- `decision` from `coordinator`: kind=approved_proposal action=integrate_patch task=a7f42075
- `review_verdict` from `critic`: verdict=approve reason=Classified as enablement_landing with specialist provenance,
- `proposal` from `coordinator`: action_name=integrate_patch
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `delegated_result` from `coordinator`: kind=baseline state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=medium log error pattern 'RuntimeError' matched 5 times
- `alert` from `robustness`: sev=high log error pattern 'RuntimeError: Engine core initialization failed' matched 1 times
- `decision` from `coordinator`: kind=approved_proposal action=baseline task=614857fe
- `review_verdict` from `critic`: verdict=approve reason=PRELUDE baseline is the natural cold-start evidence producer
- `proposal` from `orchestration`: action_name=baseline
- `delegated_result` from `coordinator`: kind=recover state=succeeded tput=None decision=None
- `delegated_result` from `coordinator`: kind=integrate_patch state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=high local server probe http://127.0.0.1:8888/health status=error
- `decision` from `coordinator`: kind=approved_proposal action=integrate_patch task=684650cc
- `review_verdict` from `critic`: verdict=advise reason=Enablement integrate_patch has specialist provenance and no 
- `proposal` from `coordinator`: action_name=integrate_patch
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=medium PRELUDE phase budget 100% consumed (325s of 324s cap)

## Degraded mode

This run executed on the **text path only**. Multimodal (image/audio) inputs were ignored, so throughput/accuracy reflect the text decoder alone. Pass `--no-allow-mm-text-fallback` to fail-fast instead.

- `Ornith-1.5-397B-FP8` (arch `Qwen3_5MoeForConditionalGeneration`): multimodal config key 'vision_config'

## External baseline (competitor target, advisory)

- Query: model=`(unset)`  gpu=`(unset)`  framework=`(any)`  precision=`(any)`  ISL/OSL=`(any)/(any)`
- Fetched at: 2026-09-15T16:19:25Z
- Status: `skipped` reason=`model_mapping_miss` (rows matched: 0)
- Warning: model name mapping miss for '/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-FP8'; no InferenceX name found

- No reference best available — orchestrator was not affected by this section.

> Advisory only. This block does not feed Objective, scoring, or any agent prompt; it is shown here purely for post-mortem comparison.
