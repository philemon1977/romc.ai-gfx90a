# Inference Optimizer Report — Qwen3.8-27B-ABLITERATED-W8A8-gdnint8_20260914T173139Z_e9102f01

- **Model**: Qwen3.8-27B-ABLITERATED-W8A8-gdnint8  (`/mnt/stripe-3mix-3t2/models/davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8`)
- **Stop reason**: `time_exhausted`
- **Why it stopped**: Wall-clock budget (--max-hours) was exhausted; the best validated result was kept.
- **⚠ Degraded mode**: ran on the TEXT path only (multimodal inputs ignored) — see 'Degraded mode' below
- **Budget**: 120 minutes
- **Generated**: 2026-09-14T19:33:50.516993+00:00

## Throughput

- baseline            : `444.9 tok/s/GPU`
- current_best        : `444.9 tok/s/GPU` (action=`baseline`)
- cumulative_gain_val : `0.00%` ⚠ never validated — nothing has promoted in this session
- ttft_mean      : `6121.4` ms
- e2el_mean      : `147098.1` ms

## Run summary

- crash_count    : 0
- pruned_families: (none)
- host           : amd-server
- platform       : AMD EPYC 7413 24-Core Processor — SMT off, NPS1 (2 NUMA nodes / 2 sockets), governor schedutil, boost on
- accelerators   : 8× unknown on the host, amdgpu 6.16.13
- stack          : aiter unknown, rocm 7.2.4, sglang unknown, vllm 0.28.0+rocm723, kernel 6.8.0-139-generic

## Event counts

- `observation`: 17
- `delegated_result`: 6
- `alert`: 5
- `event`: 4
- `heartbeat`: 3
- `decision`: 2
- `proposal`: 2
- `review_verdict`: 2

## Highlights

- `alert` from `robustness`: sev=high only 9.3min remain (<= 30min); validated_gain still 0; wind the session down now
- `alert` from `robustness`: sev=high wall-clock budget 92% consumed (elapsed=110.7min / budget=120min) and cumulative_gain_validated=0.00% — wind the session down before the deadline cuts it
- `alert` from `robustness`: sev=high agent orchestration silent for 1786s (threshold=300s)
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `alert` from `robustness`: sev=medium wall-clock budget 67% consumed with cumulative_gain_validated=0.00% — early strategy hint: cheap actions are not paying off
- `alert` from `orchestration`: sev=high Rig-property finding from research scout, not fixable by any vLLM flag: sclk pinned at 800MHz / fclk 400MHz for ALL 1748 samples across the 3766s baseline run (MI250X boost ~1.7GHz/300W; baseline drew flat 96-98W). Read-only rocm-smi --showclocks still shows DPM level 1 on every GCD. If genuinely pinned, every compute/issue-bound kernel pays ~2.1x and no config or patch lever recovers it — worth confirming DPM policy on the host before spending the remaining ~40min expecting >30% gain.
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `delegated_result` from `coordinator`: kind=specialist state=succeeded tput=None decision=None
- `delegated_result` from `coordinator`: kind=baseline state=succeeded tput=444.8566029611903 decision=None
- `delegated_result` from `coordinator`: kind=target_analysis state=succeeded tput=None decision=None
- `decision` from `coordinator`: kind=approved_proposal action=baseline task=e420b1b3
- `review_verdict` from `critic`: verdict=approve reason=baseline is exploration: its purpose is to generate the firs
- `decision` from `coordinator`: kind=approved_proposal action=target_analysis task=744cfece
- `review_verdict` from `critic`: verdict=approve reason=target_analysis is archival: it transcribes existing state w
- `proposal` from `orchestration`: action_name=baseline
- `proposal` from `orchestration`: action_name=target_analysis

## Degraded mode

This run executed on the **text path only**. Multimodal (image/audio) inputs were ignored, so throughput/accuracy reflect the text decoder alone. Pass `--no-allow-mm-text-fallback` to fail-fast instead.

- `Qwen3.8-27B-ABLITERATED-W8A8-gdnint8` (arch `Qwen3_5ForConditionalGeneration`): multimodal config key 'vision_config'

## External baseline (competitor target, advisory)

- Query: model=`(unset)`  gpu=`(unset)`  framework=`(any)`  precision=`(any)`  ISL/OSL=`(any)/(any)`
- Fetched at: 2026-09-14T17:38:03Z
- Status: `skipped` reason=`model_mapping_miss` (rows matched: 0)
- Warning: model name mapping miss for '/mnt/stripe-3mix-3t2/models/davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8'; no InferenceX name found

- No reference best available — orchestrator was not affected by this section.

> Advisory only. This block does not feed Objective, scoring, or any agent prompt; it is shown here purely for post-mortem comparison.
