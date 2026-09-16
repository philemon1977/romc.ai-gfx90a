# [Magpie/benchmark] Parent `HIP_VISIBLE_DEVICES` leaks into the ROCR-pinned child env → torch strict-check crash or "No CUDA GPUs are available"

**Labels:** `type:bug`
**Target release:** Unsure (bundled Magpie in `hyperloom-inference-optimizer==1.1.0`)
**Entry point:** Local (docker run mode)
**Optimization domain:** Inference
**Issue type:** Crash / failure
**Phase:** Setup / baseline

## Summary

When the Hyperloom harness has exported a **physical** `HIP_VISIBLE_DEVICES` (e.g. `2`) and Magpie's benchmark config overlays a **logical** `ROCR_VISIBLE_DEVICES=0` (absolute-indexed picks in multi-GPU containers), the child process inherits the contradictory pair. PyTorch's strict validation then fails at `import vllm`:

- different-length/over-index masks → `ValueError`/`RuntimeError: Engine core initialization failed` before the server ever binds a port (session logs classify it as `failure_kind=unknown`);
- same-length physical id → `torch.cuda.is_available()==False` → `No CUDA GPUs are available`.

This accounted for 100% of baseline launch failures in our gfx90a session (evidenced in `runs/baseline/*/server.log`; the server dies pre-port, so the real error is invisible to port-health logic).

## Expected behavior

Magpie's `_build_local_command` should never launch a child with an inconsistent HIP/CUDA-vs-ROCR mask pair. The InferenceX launcher (`benchmarks/vllm_mi300x.sh:63-68`) already *documents* the intended rule (derive HIP from ROCR) but only covers the case where HIP arrives unset.

## Actual behavior

`env = os.environ.copy()` keeps the inherited physical mask; the yaml overlay writes only `ROCR_VISIBLE_DEVICES`. The pair is passed through unreconciled.

## Proven fix (we ran it E2E)

A pure-insertion patch to `Magpie/modes/benchmark/benchmarker.py` reconciles HIP/CUDA to the logical `0..n-1` set that ROCR re-indexes to, with vendor guard + fail-open (inert on CUDA and on already-valid masks):

```
Device-mask reconciliation for local run: ROCR_VISIBLE_DEVICES='0' re-indexes the visible
set to 0..0; reconciled HIP_VISIBLE_DEVICES: '2' -> '0'
```
Under the patch the real serving path completes: weights load, 35/35 CUDA graphs capture, `GET /health 200 OK`.

## Suggested fix

Adopt the reconciliation in `benchmarker.py` (both `_build_local_command` and the `_execute_local_benchmark_with_reuse` path), or unset `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` whenever Magpie writes `ROCR_VISIBLE_DEVICES` for the child. We can contribute the patch as a PR if useful.

## Repro

1. In an 8-GPU container with `HIP_VISIBLE_DEVICES=0,...,7` exported to the harness process.
2. Run a baseline benchmark whose generated yaml pins `ROCR_VISIBLE_DEVICES: '0'` with `gpu_selection.auto: false` picking an absolute index ≠ 0.
3. Server dies before `/health`; `server.log` shows the mask contradiction via `torch._parse_visible_devices()`.

## Environment

MI250X (gfx90a) — but the same mechanism should reproduce on MI300-series with a leaked parent mask. vllm 0.28.0+rocm723, torch bundled in the serving env (`torch/cuda/__init__.py:833-860`).
