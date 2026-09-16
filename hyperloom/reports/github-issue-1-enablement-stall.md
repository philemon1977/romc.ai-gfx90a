# [Inference] gfx90a session: enablement patches authored & proven, but integration lane never lands them → `enablement_stalled` with 0 baselines

**Labels:** `type:bug`
**Target release:** Unsure (wheel `hyperloom-inference-optimizer==1.1.0`, installed 2026-09-14)
**Entry point:** Local (docker run mode, `HYPERLOOM_RUN_MODE=docker`)
**Optimization domain:** Inference
**Issue type:** Hang / no progress
**Phase:** Setup / baseline

## Summary

On an unsupported GPU (gfx90a / MI250X), a 3h INT8 session burned 100% of its budget in the PRELUDE enablement loop. The enablement specialist repeatedly **authored the correct fix and proved it end-to-end inside its sandbox**, but the integration lane never applied the patch to the live tree before the hard deadline, and the policy gate kept rejecting re-proposals with `enablement_round_in_flight`. Final: `stop_reason=enablement_stalled`, `baseline_tput=0.0`, `crash_count=0`.

## Expected behavior

Once an enablement round produces a proven patch, the orchestrator should integrate it and immediately re-run the blocked action (baseline) while budget remains — or at minimum yield the PRELUDE budget cap rather than looping.

## Actual behavior

- Robustness alerts: `PRELUDE phase budget consumed 1129% (3656s of 324s cap)`.
- Coordinator log: `policy_denied` ×9, top rule `enablement_round_in_flight`; orchestration `propose_action` for `baseline` repeatedly denied while an authoring round was in flight.
- Specialist round finished with the patch materialized under `reports/enablement/patches/00X_reconcile_rocm_device_masks.patch` and a sandbox E2E proof (weights loaded, 35/35 CUDA graphs, `/health 200 OK`), but `grep` of the live Magpie tree shows the patch symbol count stayed `0` until session close.
- Hard-deadline finalize fired (`sev=high: only 4.5min remain; finalize now`).

## Root cause (suspected, from session artifacts)

1. The `enablement_round_in_flight` gate serializes *everything* (including baseline re-dispatch) behind the authoring round instead of only further enablement proposals.
2. There is no deadline-aware escape hatch: when remaining budget < (integrate + one baseline), continuing authoring rounds is provably futile; the loop makes "no forward progress for several consecutive rounds" without stopping earlier (the stall detector needs several consecutive idle rounds; `orchestration silent for 696s` alerts also observed).
3. A secondary dependency gap surfaced by the fix (`/opt/venv/bin/python3` bench client missing `transformers`/`huggingface_hub`; the session's own `pip install` against default PyPI hit connection resets in a restricted-network environment) was recorded in the specialist summary as "replay on the integration lane", but no lane replayed it.

## Suggested fix

- Allow `baseline` (and other blocked actions) to run once a proven patch exists in `reports/enablement/patches/`, even mid-round, or integrate-then-retry atomically.
- Budget-aware gate: if `remaining_time < integrate + estimated_action_time`, force integration of the latest proven patch.
- Surface `setup_commands` failures of the **bench client** interpreter distinctly (PATH order: yaml puts `/opt/venv/bin` first; the serving interpreter differs from the client interpreter).

## Environment

- 8× MI250X (gfx90a), amdgpu 6.16.13, ROCm 7.2.4 container (`vllm 0.28.0+rocm723`), Ubuntu 24.04 host.
- `python3 -m hyperloom.inference_optimizer.cli optimize --framework vllm --model-class dense --tp 1 ... --no-kernel`
- Session id: `20260914T134902Z-244d9455` (model: local W8A8 INT8 Qwen3.8-27B variant, compressed-tensors)

## Additional notes

gfx90a is absent from the support matrix (MI300X/325X/355X). We accept "unsupported", but the failure mode above (proven fix never lands) would also bite supported platforms under flaky-network enablement conditions. Happy to provide the sanitized `final.md` / specialist round summaries.
