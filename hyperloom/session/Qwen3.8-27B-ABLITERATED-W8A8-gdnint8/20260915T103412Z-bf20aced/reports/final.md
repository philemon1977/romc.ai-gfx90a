# Inference Optimizer Report — Qwen3.8-27B-ABLITERATED-W8A8-gdnint8_20260915T103412Z_ebc83030

- **Model**: Qwen3.8-27B-ABLITERATED-W8A8-gdnint8  (`/mnt/stripe-3mix-3t2/models/davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8`)
- **Stop reason**: `sweep_done`
- **Why it stopped**: Post-sweep concurrency sweep did not run (disabled); the phase settled and the run closed.
- **⚠ Degraded mode**: ran on the TEXT path only (multimodal inputs ignored) — see 'Degraded mode' below
- **Budget**: 180 minutes
- **Generated**: 2026-09-15T13:28:25.868551+00:00

## Throughput

- baseline            : `512.5 tok/s/GPU`
- current_best        : `512.5 tok/s/GPU` (action=`baseline`)
- cumulative_gain_val : `0.00%` ⚠ never validated — nothing has promoted in this session
- ttft_mean      : `6438.8` ms
- e2el_mean      : `127638.6` ms

## Run summary

- crash_count    : 0
- pruned_families: (none)
- host           : amd-server
- platform       : AMD EPYC 7413 24-Core Processor — SMT off, NPS1 (2 NUMA nodes / 2 sockets), governor schedutil, boost on
- accelerators   : 8× unknown on the host, amdgpu 6.16.13
- stack          : aiter unknown, rocm 7.2.4, sglang unknown, vllm 0.28.0+rocm723, kernel 6.8.0-139-generic

## Event counts

- `observation`: 27
- `delegated_result`: 13
- `event`: 9
- `heartbeat`: 8
- `alert`: 5
- `decision`: 4
- `proposal`: 4
- `review_verdict`: 4
- `strategy_change`: 1

## Highlights

- `alert` from `robustness`: sev=medium aiter jit build_count=6 unchanged for 5 consecutive ticks; a prior hipcc invocation likely crashed mid-build
- `alert` from `robustness`: sev=high only 5.9min remain (<= 30min); validated_gain still 0; wind the session down now
- `alert` from `robustness`: sev=high wall-clock budget 97% consumed (elapsed=174.1min / budget=180min) and cumulative_gain_validated=0.00% — wind the session down before the deadline cuts it
- `decision` from `coordinator`: kind=approved_proposal action=report task=2a1f3cd9
- `review_verdict` from `critic`: verdict=approve reason=Archival report in CLOSE introduces no new measurements and 
- `decision` from `coordinator`: kind=approved_proposal action=integrate_patch task=a4c1e6fb
- `review_verdict` from `critic`: verdict=approve reason=Framework-agent enablement landing with specialist provenanc
- `proposal` from `orchestration`: action_name=report
- `proposal` from `coordinator`: action_name=integrate_patch
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `delegated_result` from `coordinator`: kind=explore state=succeeded tput=None decision=None
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `delegated_result` from `coordinator`: kind=explore state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=medium wall-clock budget 54% consumed with cumulative_gain_validated=0.00% — early strategy hint: cheap actions are not paying off
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `delegated_result` from `coordinator`: kind=explore state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=high agent orchestration silent for 3996s (threshold=300s)
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `delegated_result` from `coordinator`: kind=baseline state=succeeded tput=512.5489837765601 decision=None
- `delegated_result` from `coordinator`: kind=target_analysis state=succeeded tput=None decision=None
- `decision` from `coordinator`: kind=approved_proposal action=baseline task=51ac28ff
- `review_verdict` from `critic`: verdict=approve reason=PRELUDE baseline is an exploration framework operation that 
- `decision` from `coordinator`: kind=approved_proposal action=target_analysis task=85459fbd
- `review_verdict` from `critic`: verdict=approve reason=PRELUDE target_analysis is an archival framework operation a
- `proposal` from `orchestration`: action_name=baseline
- `proposal` from `orchestration`: action_name=target_analysis

## Degraded mode

This run executed on the **text path only**. Multimodal (image/audio) inputs were ignored, so throughput/accuracy reflect the text decoder alone. Pass `--no-allow-mm-text-fallback` to fail-fast instead.

- `Qwen3.8-27B-ABLITERATED-W8A8-gdnint8` (arch `Qwen3_5ForConditionalGeneration`): multimodal config key 'vision_config'

## External baseline (competitor target, advisory)

- Query: model=`(unset)`  gpu=`(unset)`  framework=`(any)`  precision=`(any)`  ISL/OSL=`(any)/(any)`
- Fetched at: 2026-09-15T10:34:39Z
- Status: `skipped` reason=`model_mapping_miss` (rows matched: 0)
- Warning: model name mapping miss for '/mnt/stripe-3mix-3t2/models/davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8'; no InferenceX name found

- No reference best available — orchestrator was not affected by this section.

> Advisory only. This block does not feed Objective, scoring, or any agent prompt; it is shown here purely for post-mortem comparison.
