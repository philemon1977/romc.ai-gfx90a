# Inference Optimizer Report — Ornith-1.5-35B-A3B_20260914T094955Z_a8127778

- **Model**: Ornith-1.5-35B-A3B  (`/mnt/stripe-3mix-3t2/models/ornith-ai/Ornith-1.5-35B-A3B`)
- **Stop reason**: `enablement_stalled`
- **Why it stopped**: The enablement loop made no forward progress for several consecutive rounds and stopped instead of re-deriving the same fix.
- **⚠ Degraded mode**: ran on the TEXT path only (multimodal inputs ignored) — see 'Degraded mode' below
- **Budget**: 120 minutes
- **Generated**: 2026-09-14T12:45:04.441589+00:00

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

- `heartbeat`: 342
- `alert`: 178
- `observation`: 51
- `delegated_result`: 8
- `decision`: 7
- `proposal`: 7
- `review_verdict`: 7
- `lease_expired`: 5
- `advice`: 3
- `event`: 3
- `strategy_change`: 1

## Highlights

- `alert` from `robustness`: sev=medium PRELUDE phase budget 3976% consumed (8587s of 216s cap)
- `alert` from `robustness`: sev=medium log error pattern 'RuntimeError' matched 3 times
- `alert` from `robustness`: sev=high log error pattern 'RuntimeError: Engine core initialization failed' matched 1 times
- `alert` from `robustness`: sev=high orchestration stalled for 150 ticks (threshold=15); last progress at tick 2
- `alert` from `robustness`: sev=medium agent coordinator hit policy_denied 10 times (>= 3); top rule=enablement_round_in_flight
- `alert` from `robustness`: sev=medium aiter jit build_count=6 unchanged for 150 consecutive ticks; a prior hipcc invocation likely crashed mid-build
- `alert` from `robustness`: sev=high only 6.2min remain (<= 30min); validated_gain still 0; wind the session down now
- `alert` from `robustness`: sev=high wall-clock budget 118% consumed (elapsed=142.0min / budget=120min) and cumulative_gain_validated=0.00% — wind the session down before the deadline cuts it
- `decision` from `coordinator`: kind=approved_proposal action=integrate_patch task=313e6341
- `review_verdict` from `critic`: verdict=approve reason=Enablement-tagged integrate_patch has specialist provenance 
- `proposal` from `coordinator`: action_name=integrate_patch
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=medium PRELUDE phase budget 2466% consumed (5326s of 216s cap)
- `delegated_result` from `coordinator`: kind=baseline state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=medium log error pattern 'RuntimeError' matched 4 times
- `alert` from `robustness`: sev=high log error pattern 'RuntimeError: Engine core initialization failed' matched 2 times
- `alert` from `robustness`: sev=high orchestration stalled for 145 ticks (threshold=15); last progress at tick 2
- `decision` from `coordinator`: kind=approved_proposal action=baseline task=3f5caae8
- `review_verdict` from `critic`: verdict=approve reason=Baseline in PRELUDE is the natural exploration action after 
- `proposal` from `orchestration`: action_name=baseline
- `delegated_result` from `coordinator`: kind=integrate_patch state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=medium wall-clock budget 72% consumed with cumulative_gain_validated=0.00% — consider escalating strategy before the wind-down
- `alert` from `robustness`: sev=high agent orchestration silent for 3126s (threshold=300s)
- `decision` from `coordinator`: kind=approved_proposal action=integrate_patch task=7eb609d6
- `review_verdict` from `critic`: verdict=approve reason=This is an enablement landing with specialist provenance and
- `proposal` from `coordinator`: action_name=integrate_patch
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=medium agent coordinator hit policy_denied 6 times (>= 3); top rule=enablement_round_in_flight
- `alert` from `robustness`: sev=medium aiter jit build_count=6 unchanged for 145 consecutive ticks; a prior hipcc invocation likely crashed mid-build
- `alert` from `orchestration`: sev=high CORRECTION: gate holder d00cdedf is NOT phantom — its underlying enablement-specialist PROCESS is alive and writing under run dir 78b25f2832ff45ff949ad9c0e9634cf3 (process.log mtime 10:20:24Z, scratch/server3.log mtime 10:22Z, launcher3.out 10:17, patches/ touched 10:19, triton Worker_TP1 pid=18890 collecting artifacts). Task registry (get_running_tasks) shows EMPTY => registry desynced from live process, which is why no lease/force-reap ever fires. Live evidence window now: server3.log frozen ~33min (10:22 -> 10:55Z), consistent with the prior wedged pattern. ACTION FOR ROBUSTNESS: do not wait for TTL (lapsed); locate pid group writing /session/.../runs/specialist/78b25f2832ff45ff949ad9c0e9634cf3/scratch/server3.log (fuser/lsof on that file), kill the tree, then clear the enablement_round record so PolicyGate rearms. Baseline probe (round-9, idempotency baseline-round-9, params unchanged: unset CUDA/HIP_VISIBLE_DEVICES, pre-kill orphans 18501/18709) re-fires on gate clear; I will stop probing and emit skip-style escalation with this same evidence if the block persists past ~11:25Z.
- `alert` from `robustness`: sev=medium PRELUDE phase budget 932% consumed (2013s of 216s cap)
- `alert` from `robustness`: sev=medium log error pattern 'RuntimeError' matched 1 times
- `alert` from `robustness`: sev=high log error pattern 'RuntimeError: Engine core initialization failed' matched 1 times
- `alert` from `robustness`: sev=high orchestration stalled for 140 ticks (threshold=15); last progress at tick 2
- `alert` from `robustness`: sev=medium agent coordinator hit policy_denied 5 times (>= 3); top rule=enablement_round_in_flight
- `alert` from `robustness`: sev=medium aiter jit build_count=6 unchanged for 140 consecutive ticks; a prior hipcc invocation likely crashed mid-build
- `alert` from `robustness`: sev=medium PRELUDE phase budget 902% consumed (1948s of 216s cap)
- `alert` from `robustness`: sev=medium log error pattern 'RuntimeError' matched 1 times
- `alert` from `robustness`: sev=high log error pattern 'RuntimeError: Engine core initialization failed' matched 1 times
- `alert` from `robustness`: sev=high orchestration stalled for 135 ticks (threshold=15); last progress at tick 2
- `alert` from `robustness`: sev=medium agent coordinator hit policy_denied 4 times (>= 3); top rule=enablement_round_in_flight
- `alert` from `robustness`: sev=medium aiter jit build_count=6 unchanged for 135 consecutive ticks; a prior hipcc invocation likely crashed mid-build
- `alert` from `robustness`: sev=medium PRELUDE phase budget 881% consumed (1903s of 216s cap)
- `alert` from `robustness`: sev=medium log error pattern 'RuntimeError' matched 1 times
- `alert` from `robustness`: sev=high log error pattern 'RuntimeError: Engine core initialization failed' matched 1 times
- `alert` from `robustness`: sev=high orchestration stalled for 130 ticks (threshold=15); last progress at tick 2
- `alert` from `robustness`: sev=medium agent coordinator hit policy_denied 3 times (>= 3); top rule=enablement_round_in_flight
- `alert` from `robustness`: sev=medium aiter jit build_count=6 unchanged for 130 consecutive ticks; a prior hipcc invocation likely crashed mid-build
- `alert` from `robustness`: sev=medium PRELUDE phase budget 857% consumed (1851s of 216s cap)
- `alert` from `orchestration`: sev=high Enablement gate is holding on a phantom task: baseline-round-4 (d00cdedf...) and baseline-round-5 (d00cdedf...) both DENIED rule=enablement_round_in_flight, but get_running_tasks returns ZERO in-flight tasks on the same ticks. Previous holder 78b25f28 is gone; d00cdedf never appeared in the registry. This is gate/registry desync in PolicyGate state, not a live round — retrying baseline cannot clear it. Robustness action ladder needed: force-reap/clear enablement_round state for task d00cdedfc173451eaf5b312d08e00da9 (and any lingering enablement authoring records). The 'orchestration stalled 125 ticks' and pid=15383 Engine-core-init alerts are downstream artifacts of this same block (attempt-1 stale log). Until cleared, every PRELUDE tick is wasted; ~117min session budget remains and baseline is fully prepared to fire.

## Degraded mode

This run executed on the **text path only**. Multimodal (image/audio) inputs were ignored, so throughput/accuracy reflect the text decoder alone. Pass `--no-allow-mm-text-fallback` to fail-fast instead.

- `Ornith-1.5-35B-A3B` (arch `Qwen3_5MoeForConditionalGeneration`): multimodal config key 'vision_config'

## External baseline (competitor target, advisory)

- Query: model=`(unset)`  gpu=`(unset)`  framework=`(any)`  precision=`(any)`  ISL/OSL=`(any)/(any)`
- Fetched at: 2026-09-14T10:37:19Z
- Status: `skipped` reason=`model_mapping_miss` (rows matched: 0)
- Warning: model name mapping miss for '/mnt/stripe-3mix-3t2/models/ornith-ai/Ornith-1.5-35B-A3B'; no InferenceX name found

- No reference best available — orchestrator was not affected by this section.

> Advisory only. This block does not feed Objective, scoring, or any agent prompt; it is shown here purely for post-mortem comparison.
