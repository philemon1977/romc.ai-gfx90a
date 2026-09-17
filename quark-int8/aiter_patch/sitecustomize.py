"""Surgical patch: enable vLLM's AITER **INT8 scaled-MM linear** path on CDNA2
(gfx90a / MI250X).

Background: `vllm._aiter_ops.is_aiter_found_and_supported()` requires
`get_cdna_version() > 2`, so on MI250X:
  * the custom op `vllm.rocm_aiter_w8a8_gemm` is never registered (its
    registration lives inside that gate), and
  * `rocm_aiter_ops.is_linear_enabled()` returns None, so
    `AiterInt8ScaledMMLinearKernel.is_supported()` reports "requires setting
    VLLM_ROCM_USE_AITER=1..." and vLLM always uses TritonInt8ScaledMMLinearKernel.
The AITER CK int8 GEMM itself *does* build for gfx90a and is numerically correct
(measured max_rel_err 0.0019 for M256/N512/K4096 on gfx90a), so the gate is
conservative rather than a hardware limitation.

This patch therefore only:
  1. registers `vllm.rocm_aiter_w8a8_gemm` when it is missing, and
  2. makes `is_linear_enabled()` perform the env-var check the gate wraps.
Every other aiter feature (fused MoE, attention, rmsnorm, rope, ...) keeps its
upstream gating, so unvalidated gfx90a kernels stay off.

Implemented as a lazy import hook: importing this file never imports vLLM itself
(doing so early silently disables vLLM's INFO logging).
"""
import importlib.abc
import os
import sys

TARGET = "vllm._aiter_ops"
_ENV_ON = ("1", "true", "yes", "on")


def _enabled() -> bool:
    return (os.environ.get("VLLM_ROCM_USE_AITER", "0").lower() in _ENV_ON
            and os.environ.get("VLLM_ROCM_USE_AITER_LINEAR", "1").lower() in _ENV_ON)


def _apply(module) -> None:
    import torch

    # 1) register the INT8 GEMM custom op if the gate skipped it
    registered = hasattr(torch.ops.vllm, "rocm_aiter_w8a8_gemm")
    if not registered:
        from vllm.utils.torch_utils import direct_register_custom_op

        direct_register_custom_op(
            op_name="rocm_aiter_w8a8_gemm",
            op_func=module._rocm_aiter_w8a8_gemm_impl,
            fake_impl=module._rocm_aiter_w8a8_gemm_fake,
        )
        print("[gfx90a-patch] registered vllm.rocm_aiter_w8a8_gemm (upstream gate "
              "skipped it on CDNA2)", flush=True)

    # 2) is_linear_enabled(): upstream is `cls._AITER_ENABLED and cls._LINEAR_ENABLED`
    #    behind the CDNA3+ gate -> restore exactly that logic without the gate.
    cls = module.rocm_aiter_ops

    def is_linear_enabled(klass) -> bool:
        return bool(klass._AITER_ENABLED and klass._LINEAR_ENABLED)

    cls.is_linear_enabled = classmethod(is_linear_enabled)
    print("[gfx90a-patch] rocm_aiter_ops.is_linear_enabled restored on CDNA2", flush=True)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        for finder in list(sys.meta_path):
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                continue
            orig_exec = spec.loader.exec_module

            def exec_module(mod, _orig=orig_exec):
                _orig(mod)
                try:
                    _apply(mod)
                except Exception as exc:  # never break model loading
                    print(f"[gfx90a-patch] patch failed: {exc!r}", flush=True)

            spec.loader.exec_module = exec_module
            return spec
        return None


if _enabled():
    sys.meta_path.insert(0, _PatchFinder())
