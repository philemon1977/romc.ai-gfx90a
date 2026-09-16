# [Orchestrator] Two recovery defects: conversational roles stall at `max_turns=12`; kill+resume leaves phantom coordinator leases that deadlock IR-1

**Labels:** `type:bug`
**Target release:** Unsure (`hyperloom-inference-optimizer==1.1.0`)
**Entry point:** Local (docker run mode)
**Optimization domain:** Inference
**Issue type:** Hang / no progress
**Phase:** Optimization loop

## Part A — orchestration role deadlocks at the 12-turn cap

With the `claude-agent-sdk` backend (any backend, actually), the orchestration role's conversation hit `Reached maximum number of turns (12)` and the whole tick loop went silent (asyncio select idle, 0 children alive; robustness alert: `orchestration silent for 696s`). The session was alive but brain-dead — no proposals, no integration, no stop-reason.

We locally raised `_CONVERSATIONAL_MIN_MAX_TURNS` from 12 to 36 in `hyperloom/orchestrator/roles/claude.py:132` and ticks resumed advancing. Two questions:

1. 12 seems too low for enablement-heavy PRELUDE rounds (the role consumes turns on tool-call chains before it can even emit an action).
2. When a role does exhaust its turn budget, the loop should recover (fresh conversation carrying a summary, or raise a stop_reason) rather than idle until the hard deadline.

## Part B — `--resume-from` after an external kill inherits phantom leases

If the optimize process is killed (OOM/keyboard/crash) while a specialist holds `resource_lock` leases (`benchmark_lane`, `server_lifecycle`, `gpu_research_lane`, `research_lane`, `profile_lane`, plus `gpu_leases` rows), the resumed session can deadlock: the coordinator sees `phase=enablement_stalled`-style gates blocked by leases whose holders are dead, while the dispatcher's reaper only runs *after* the first tick — which the IR-1 GPU-idle gate never lets arrive.

Observed: `dispatcher: reclaimed 1 running task(s) with dead holders` fires late (≈90s after first tick) in one resume and never in another; the resume session spun at tick parity until deadline.

**Manual recovery recipe that worked** (please productize — e.g. verify lease holders' PIDs at session load, or lease-on-PID-boot-time):

```sql
UPDATE tasks SET status='failed' WHERE status IN ('running','leased');
UPDATE leases SET expires_at=<past> ; UPDATE gpu_leases SET expires_at=<past>;
```

## Environment

8× MI250X (gfx90a) docker run-mode; single-container, all 8 GPUs visible; vllm 0.28.0. Both parts reproduced twice in one 3h budget on session `20260914T134902Z-244d9455`; sanitized logs available.
