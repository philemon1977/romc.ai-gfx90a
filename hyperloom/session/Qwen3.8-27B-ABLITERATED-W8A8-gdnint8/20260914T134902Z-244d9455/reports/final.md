# Inference Optimizer Report — Qwen3.8-27B-ABLITERATED-W8A8-gdnint8_20260914T134902Z_5ae731fb

- **Model**: Qwen3.8-27B-ABLITERATED-W8A8-gdnint8  (`/mnt/stripe-3mix-3t2/models/davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8`)
- **Stop reason**: `enablement_stalled`
- **Why it stopped**: The enablement loop made no forward progress for several consecutive rounds and stopped instead of re-deriving the same fix.
- **⚠ Degraded mode**: ran on the TEXT path only (multimodal inputs ignored) — see 'Degraded mode' below
- **Budget**: 180 minutes
- **Generated**: 2026-09-14T16:45:25.197251+00:00

## Throughput

- baseline            : `0.0 tok/s/GPU`
- cumulative_gain_val : `0.00%` ⚠ never validated — nothing has promoted in this session

## Run summary

- crash_count    : 0
- pruned_families: (none)
- host           : amd-server
- platform       : AMD EPYC 7413 24-Core Processor — SMT off, NPS1 (2 NUMA nodes / 2 sockets), governor schedutil, boost on
- accelerators   : 8× unknown on the host, amdgpu 6.16.13
- stack          : aiter unknown, rocm 7.2.4, sglang unknown, vllm 0.28.0+rocm723, kernel 6.8.0-139-generic

## Event counts

- `observation`: 47
- `heartbeat`: 24
- `alert`: 18
- `delegated_result`: 9
- `decision`: 7
- `proposal`: 7
- `review_verdict`: 7
- `lease_expired`: 5
- `event`: 4
- `advice`: 3
- `strategy_change`: 2

## Highlights

- `alert` from `robustness`: sev=medium PRELUDE phase budget 718% consumed (2326s of 324s cap)
- `alert` from `robustness`: sev=high log error pattern 'RuntimeError: Engine core initialization failed' matched 1 times
- `alert` from `robustness`: sev=medium agent coordinator hit policy_denied 9 times (>= 3); top rule=enablement_round_in_flight
- `alert` from `robustness`: sev=medium aiter jit build_count=6 unchanged for 16 consecutive ticks; a prior hipcc invocation likely crashed mid-build
- `alert` from `robustness`: sev=medium log error pattern 'RuntimeError' matched 1 times
- `alert` from `robustness`: sev=medium log error pattern 'Address already in use' matched 1 times
- `alert` from `robustness`: sev=high hard deadline cutoff: only 4.5min remain (<= 5min); finalize now
- `decision` from `coordinator`: kind=approved_proposal action=integrate_patch task=118f40f2
- `review_verdict` from `critic`: verdict=advise reason=Enablement landing is structurally eligible: specialist prov
- `proposal` from `coordinator`: action_name=integrate_patch
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=medium PRELUDE phase budget 107% consumed (345s of 324s cap)
- `alert` from `robustness`: sev=high log error pattern 'RuntimeError: Engine core initialization failed' matched 1 times
- `delegated_result` from `coordinator`: kind=baseline state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=medium agent coordinator hit policy_denied 5 times (>= 3); top rule=enablement_round_in_flight
- `alert` from `robustness`: sev=medium aiter jit build_count=6 unchanged for 10 consecutive ticks; a prior hipcc invocation likely crashed mid-build
- `decision` from `coordinator`: kind=approved_proposal action=baseline task=0668d71b
- `review_verdict` from `critic`: verdict=approve reason=Baseline is a PRELUDE exploration/framework_op action and is
- `proposal` from `orchestration`: action_name=baseline
- `delegated_result` from `coordinator`: kind=integrate_patch state=succeeded tput=None decision=None
- `decision` from `coordinator`: kind=approved_proposal action=integrate_patch task=34d103b9
- `review_verdict` from `critic`: verdict=advise reason=Enablement patch has specialist provenance and no contradict
- `proposal` from `coordinator`: action_name=integrate_patch
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=medium log error pattern 'RuntimeError' matched 2 times
- `alert` from `robustness`: sev=medium log error pattern 'Address already in use' matched 1 times
- `delegated_result` from `coordinator`: kind=baseline state=succeeded tput=None decision=None
- `decision` from `coordinator`: kind=approved_proposal action=baseline task=48e0ae2b
- `review_verdict` from `critic`: verdict=approve reason=PRELUDE baseline is an exploration/framework action that gen
- `proposal` from `orchestration`: action_name=baseline
- `delegated_result` from `coordinator`: kind=integrate_patch state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=medium PRELUDE phase budget 1129% consumed (3656s of 324s cap)
- `decision` from `coordinator`: kind=approved_proposal action=integrate_patch task=90f715c0
- `review_verdict` from `critic`: verdict=approve reason=Classified as enablement_landing/framework-agent integrate_p
- `proposal` from `coordinator`: action_name=integrate_patch
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=medium agent coordinator hit policy_denied 3 times (>= 3); top rule=enablement_round_in_flight
- `alert` from `robustness`: sev=medium aiter jit build_count=6 unchanged for 5 consecutive ticks; a prior hipcc invocation likely crashed mid-build
- `alert` from `robustness`: sev=medium agent orchestration silent for 696s (threshold=300s)
- `alert` from `robustness`: sev=medium log error pattern 'RuntimeError' matched 2 times
- `delegated_result` from `coordinator`: kind=baseline state=succeeded tput=None decision=None
- `delegated_result` from `coordinator`: kind=target_analysis state=succeeded tput=None decision=None
- `decision` from `coordinator`: kind=approved_proposal action=target_analysis task=c640a9ee
- `review_verdict` from `critic`: verdict=approve reason=target_analysis is archival/framework_op; it only transcribe
- `decision` from `coordinator`: kind=approved_proposal action=baseline task=2fc18296
- `review_verdict` from `critic`: verdict=approve reason=PRELUDE baseline is an exploration/framework_op action; the 
- `proposal` from `orchestration`: action_name=target_analysis
- `proposal` from `orchestration`: action_name=baseline

## Degraded mode

This run executed on the **text path only**. Multimodal (image/audio) inputs were ignored, so throughput/accuracy reflect the text decoder alone. Pass `--no-allow-mm-text-fallback` to fail-fast instead.

- `Qwen3.8-27B-ABLITERATED-W8A8-gdnint8` (arch `Qwen3_5ForConditionalGeneration`): multimodal config key 'vision_config'

## External baseline (competitor target, advisory)

- Query: model=`(unset)`  gpu=`(unset)`  framework=`(any)`  precision=`(any)`  ISL/OSL=`(any)/(any)`
- Fetched at: 2026-09-14T13:49:44Z
- Status: `skipped` reason=`model_mapping_miss` (rows matched: 0)
- Warning: model name mapping miss for '/mnt/stripe-3mix-3t2/models/davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8'; no InferenceX name found

- No reference best available — orchestrator was not affected by this section.

> Advisory only. This block does not feed Objective, scoring, or any agent prompt; it is shown here purely for post-mortem comparison.
