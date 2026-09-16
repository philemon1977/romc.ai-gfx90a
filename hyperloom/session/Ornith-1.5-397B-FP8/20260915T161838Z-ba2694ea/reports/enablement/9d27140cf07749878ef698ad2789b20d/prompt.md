## 0. MANDATE

- deliverable: a source patch and/or up to 6 ranked config variants addressing the gap below
- anchor: `gap.enablement.unknown`

Run status (read-only context; do NOT re-state these as your own measurements):
- KEEP threshold this cycle: 1.00%

Judged by: the Coordinator benches your proposals end-to-end against
the sealed baseline and decides KEEP/REVERT; the accuracy gate runs
alongside. You are not asked to prove the number.

## 2. HARDWARE CONTEXT

- gpu_type: mi250x
- allocated specialist GPU ids: 0, 1, 2, 3, 4, 5, 6, 7
- TP: 8

Workload:
- precision: fp8
- concurrency: 64
- ISL (input seq len): 1024
- OSL (output seq len): 1024
- max_model_len: 6144

## 2a. EXECUTION BUDGET (wall-clock)

- Hard wall-clock budget for this entire dispatch: **3600s (~60 min)**.
- Dispatch started at: 2026-09-15T16:24:07.856943+00:00 (UTC).
- The Coordinator hard-kills your subprocess when this budget is exhausted — turns are NOT the stop signal. Scope your work to reach a deliverable conclusion inside the budget.
- Self-throttle: check elapsed wall-clock with Bash (``date -u +%s`` vs the start above), keep your ``specialist_done.partial.json`` checkpoint current, and write the final ``specialist_done.json`` before the budget runs out so your best work is never lost to a kill.

## 3. GAP STATEMENT

- gap_canonical_id: `gap.enablement.unknown`
- layer: framework
- symptom: vllm cannot launch Ornith-1.5-397B-FP8: unknown

Most recent evidence:
```json
{
  "failure_kind": "unknown",
  "model": "Ornith-1.5-397B-FP8"
}
```

## 1b. ENABLEMENT PLAYBOOK

GOAL: make model `Ornith-1.5-397B-FP8` run correctly under the `vllm` backend. It currently fails to start, or it starts but fails its accuracy eval.

FAILURE CLASS: unknown (confidence 0.00).
ERROR EXCERPT: vllm cannot launch Ornith-1.5-397B-FP8: unknown

ALLOWED SOURCE ROOTS (code edits outside these are rejected):
  - /opt/envs/vllm/lib/python3.12/site-packages/
  - /opt/envs/vllm/lib/python3.12/site-packages/aiter/
  - /opt/envs/vllm/lib/python3.12/site-packages/aiter_meta/
  - /opt/envs/vllm/lib/python3.12/site-packages/vllm/
  - /opt/envs/vllm/lib/python3.1/site-packages/aiter/
  - /opt/envs/vllm/lib/python3.1/site-packages/aiter_meta/
  - /opt/envs/vllm/lib/python3.1/site-packages/vllm/
  - /opt/rocm/
  - (discovery summary: atom=missing vllm=ok sglang=missing aiter=ok xdit=missing custom=missing)
  - (vllm installed version: 0.28.0+rocm723)
  - the ROCm / HIP / aiter source tree (/opt/rocm, aiter)
  - the serving-framework source tree (e.g. sglang / vllm / atom)

ENABLEMENT METHODOLOGY (advisory — you decide how to apply it):

DIAGNOSE ONCE, THEN CLIMB ONLY AS FAR AS NEEDED. Enablement has two axes:
  - Diagnosis: work out WHICH capability layer is missing. Read the failure signature below, read the model's config.json architecture, check the framework's supported-architecture registry and installed version, and check upstream (WebSearch / mcp__pr_monitor__*) whether the capability already exists and in which version/PR. This picks your ENTRY rung.
  - Climb: start at the LOWEST plausible rung and go up only when the current rung cannot make it boot. A model whose architecture is already supported but merely un-wired needs only the cheap top rungs (a flag / a small patch) — do NOT pull code or compile for it. A genuinely-new architecture climbs higher.
After each cleared boot failure, RE-DIAGNOSE the new (deeper) failure and pick a rung again — enablement is serial and progress is stacked.

THE LADDER (increasing complexity — enter at the lowest rung that fits):
  - Rung 0 - Diagnose / capability-gap localization (read-only): classify the failure, read config.json, check the supported-arch registry + version, look up upstream. Output: the missing layer and your chosen entry rung.
  - Rung 1 - Serve-flag / config wire-up: the architecture is supported and only a serve flag / env / tokenizer-mode / trivial registration alias is missing. No new code or dependencies.
  - Rung 2 - In-tree source patch: a unified diff against the INSTALLED source tree — register the arch, a small forward/config/tokenizer bridge, or backport a merged PR. Pure Python, no compile.
  - Rung 3 - Attempt-scoped runtime: the capability lives in a DIFFERENT version — acquire a wheel / editable checkout / ref into an isolated per-attempt venv. No compile, no shared-venv mutation (record it in setup_commands).
  - Rung 4 - Source localization: localize a merged-PR / vendored closure into the source root; changes touching compiled or build-backend files defer to Rung 5.
  - Rung 5 - Off-loop compiled build: AITER / sgl-kernel / vLLM-from-source, built in an isolated venv with pinned ROCm torch constraints on the off-loop build lane. Request it via needs_targeted_build (see below); do not compile inline.

FAILURE-KIND -> RECOMMENDED ENTRY RUNG (advisory, not a hard mapping):
  - serve_flag / tokenizer_error -> Rung 1
  - missing_model_arch (pure registration) / capability_disabled -> Rung 2
  - missing_model_arch / missing_weight (absent here, present in another version) -> Rung 3
  - import_error / merged-PR closure -> Rung 4
  - hip_kernel_missing / native unsupported_dtype / missing compiled symbol -> Rung 5
  - resource_constraint (OOM / GPU count) -> NOT a code gap; cannot be patched
  - accuracy_below_floor / eval_generation_pathology / eval_runtime_failure -> re-diagnose against the failing eval contract (answer quality, or generation that never terminates), then enter at the rung the underlying gap implies

This failure classified as `unknown` — use the table above to pick your entry rung.

ENVIRONMENT SETUP (installs are allowed AND must be recorded):
  - You MAY install missing/stale packages or CLI tools when that is what the model needs to build or run — e.g. `pip install -U transformers`, `pip install <dep>`, `apt-get install -y gh`, `npm install -g <tool>`. Use non-interactive, version-pinned commands where possible.
  - For EVERY install/setup command you rely on, record it VERBATIM in the `setup_commands` list of your final `specialist_done` (a JSON array of shell strings). integrate_patch replays these (allowlisted) before applying your patch and booting, so an install you depended on is reproduced rather than lost after your session ends. An unrecorded install will NOT persist.
  - Keep setup commands minimal and deterministic (pin versions), non-interactive (`-y` / `--yes`), and limited to package/tool installation — they are validated against an install-only allowlist on replay.
  - If NO environment setup is needed (a pure source fix), leave `setup_commands` empty.

PROGRESS DELIVERABLE (serial enablement — advancing the boot one step counts):
  - INCREMENTAL PROGRESS IS A FIRST-CLASS DELIVERABLE. Enablement gaps are serial: clearing one boot failure usually reveals a deeper one. You do NOT have to reach full end-to-end runnability in this one budget window.
  - If you cannot make the combo fully run, apply the SMALLEST CHANGE that ADVANCES the boot PAST THE CURRENT failure — clear THIS error even if a new, different failure then appears. The change is KEPT and stacked as a base; the next round resumes from the deeper failure. One step forward is strictly better than returning nothing. The change may be a source patch, a serve flag, an env var, or a dependency install — whichever is simplest.
  - Record the change: a source patch in ``patches_written``, serve-flag or env-var changes in ``proposal_set`` (each entry as ``extra_server_args`` or ``extra_envs``), dependency installs in ``setup_commands``. Set ``empty=false`` and in ``summary`` state which failure you cleared and what the next (deeper) failure now is.
  - Return ``empty=true`` ONLY when you cannot advance past the CURRENT failure by even one step — NOT merely because full runnability is out of reach this round.

TARGETED BUILD (request a compiled / from-source component when a patch cannot deliver it):
  - REQUESTING A COMPILED / FROM-SOURCE BUILD. If clearing this gap needs a *compiled* component (a new AITER FP4/MLA/NSA op, sgl-kernel) or a from-source framework build (e.g. a newer vLLM that NATIVELY implements this architecture, which a source patch against the INSTALLED tree cannot provide), do NOT fake it with an install command or a stub patch. Emit a ``needs_targeted_build`` object in your final ``specialist_done`` and the Coordinator runs it off-loop on an isolated, ROCm-safe build lane.
  - ``needs_targeted_build`` schema: ``{component, capability, repo_url, ref, reason}``. ``component`` is one of ``aiter`` / ``sgl_kernel`` / ``vllm_source`` / ``framework_ext``. ``capability`` names the missing op / arch (e.g. ``deepseek_v4_nsa`` / ``fp4_moe``). ``repo_url`` + ``ref`` are OPTIONAL but HIGH-VALUE: if you found (via WebSearch / mcp__pr_monitor__*) a specific upstream PR / tag / commit that implements the fix, name it (a GitHub PR URL, ``PR:1234``, a tag, or a sha) so the build checks out exactly that; leave them empty for tag-descending autoselect. ``reason`` is a one-line evidence summary.
  - A build request is COMPLEMENTARY to a source patch, not a replacement: you MAY both author the smallest patch that advances the boot one step AND request a build for the compiled/from-source piece the patch cannot cover. Setting ``needs_targeted_build`` counts as a real deliverable — do NOT set ``empty=true`` when you emit one.

HEURISTICS (judgment calls worth weighing before you reach for a fix):
  - `--enforce-eager` / `--disable-cuda-graph` (or any equivalent force-eager flag) is a DANGEROUS lever. Disabling graph capture papers over many unrelated failures, but it also silently changes the runtime path and can cause the baseline itself to regress or behave abnormally (different latency/throughput profile, masked kernel issues) relative to the graphed path. Treat it as a measure of LAST RESORT with a narrow, limited scope — reach for it only when the combo genuinely cannot boot AT ALL with graph capture enabled — never as a default, a convenience shortcut, or a way to quietly make an eval pass.
  - SEARCH FOR THE OFFICIAL LAUNCH COMMAND BEFORE AUTHORING FROM SCRATCH. You can read the current framework (e.g. sglang / vllm), the model architecture, and the GPU type from the task context — use WebSearch (or mcp__pr_monitor__*) to find the framework's or model vendor's own recommended serve command for that (model, backend, GPU) combination (official docs, model card, launch scripts, GitHub examples/issues). Mine it for the right flags, env vars, dtype, and parallelism settings instead of guessing blind. This is an important step that keeps you from reinventing the wheel behind closed doors.

INVARIANTS:
  - If the fix requires a source edit, the patch MUST be a valid unified diff that applies cleanly (`git apply --check` must pass) against the live source tree.  A serve-flag, env-var, or dependency-install fix requires no patch at all — set ``patches_written: []`` and record the change in ``proposal_set`` (for env/flag) or ``setup_commands`` (for installs).
  - Only *source edits* must stay under the allowed source roots listed below; touching any other path with a code patch is a hard reject. (Environment setup via ENVIRONMENT SETUP below is separate and allowed.)
  - Do NOT fabricate throughput/latency/accuracy numbers, and do NOT alter the eval dataset/task/metric/limit or the result parsing to inflate a score — the gate here is RUNNABILITY (server boots + minimal inference) or, for an eval-origin round, the real model output meeting the accuracy floor; not perf.
  - Prefer the smallest bridging change that makes the combo run correctly — or, when that is out of reach this round, the SMALLEST CHANGE (patch, serve flag, env var, or install) that ADVANCES past the current failure (a deeper boot gap, or a real accuracy gain toward the floor) (see PROGRESS DELIVERABLE below); do not refactor unrelated code.
  - If a discovered PR already implements the fix, adapt/backport it rather than authoring from scratch.

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
