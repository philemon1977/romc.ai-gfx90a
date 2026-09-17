# SPDX-License-Identifier: Apache-2.0
"""Lazy sitecustomize that routes vLLM's WNA16 MoE through the MI250X GEMV kernel.

Follows the project's established pattern (patches/gfx90a: a module + a gate env var +
a monkeypatch installed from the outside, never editing site-packages).

Hook detail: vllm/model_executor/layers/fused_moe/experts/triton_moe.py imports the helper
BY NAME (line 24), so patching fused_moe.invoke_fused_moe_wna16_triton_kernel would not be
seen; we patch the name in the triton_moe module namespace instead.
This file imports nothing from vllm at import time (doing so early breaks vLLM's logging).
"""
import importlib.abc
import os
import sys

TARGET = "vllm.model_executor.layers.fused_moe.experts.triton_moe"
_ENV_ON = ("1", "true", "yes", "on")


def _enabled() -> bool:
    return os.environ.get("MI250_MOE_GEMV", "0").lower() in _ENV_ON


def _apply(module) -> None:
    # 内核模块可选：mi250_moe_gemv_gs 是 group_size 泛化版（由 block_shape[1] 决定），
    # 默认用它以便同时支持 gs=128 (Ornith) 与 gs=32 (DSV4.1 CT-int4)。
    import importlib
    _modname = os.environ.get("MI250_MOE_GEMV_MODULE", "mi250_moe_gemv_gs")
    _m = importlib.import_module(_modname)
    invoke_gemv_wna16 = _m.invoke_gemv_wna16
    print(f"[MI250_MOE_GEMV] kernel module = {_modname} "
          f"(gs from block_shape: {'generic' if _modname.endswith('_gs') else 'fixed 128'})",
          flush=True)
    _m._orig_call = module.invoke_fused_moe_wna16_triton_kernel

    # ★ 关键：apply() 的签名里有 topk_ids（第 6 个位置参数），暂存下来给我们的内核直接用，
    #   从而不必从 sorted_token_ids/expert_ids 反推专家（那一层是上一次失败的原因）。
    cls = getattr(module, "TritonWNA16Experts", None)
    if cls is not None and not getattr(cls, "_mi250_gemv_wrapped", False):
        orig_apply = cls.apply

        def apply_patched(self, *args, **kwargs):
            ids = kwargs.get("topk_ids", args[5] if len(args) > 5 else None)
            _m.set_current_topk(ids)
            try:
                return orig_apply(self, *args, **kwargs)
            finally:
                _m.clear_current_topk()

        cls.apply = apply_patched
        cls._mi250_gemv_wrapped = True
        print("[MI250_MOE_GEMV] patched TritonWNA16Experts.apply (stash topk_ids)", flush=True)

    orig = module.invoke_fused_moe_wna16_triton_kernel

    def patched(A, B, C, B_scale, B_zp, topk_weights, sorted_token_ids, expert_ids,
                num_tokens_post_padded, mul_routed_weight, top_k, config, compute_type,
                use_int8_w8a16, use_int4_w4a16, block_shape=None):
        try:
            if invoke_gemv_wna16(A, B, C, B_scale, B_zp, topk_weights, sorted_token_ids,
                                 expert_ids, num_tokens_post_padded, mul_routed_weight,
                                 top_k, config, compute_type, use_int8_w8a16,
                                 use_int4_w4a16, block_shape):
                return
        except Exception as exc:      # never break the model: fall back to upstream
            global _warned
            if not _warned:
                _warned = True
                print(f"[MI250_MOE_GEMV] falling back to upstream kernel: {exc!r}", flush=True)
        return orig(A, B, C, B_scale, B_zp, topk_weights, sorted_token_ids, expert_ids,
                    num_tokens_post_padded, mul_routed_weight, top_k, config, compute_type,
                    use_int8_w8a16, use_int4_w4a16, block_shape)

    module.invoke_fused_moe_wna16_triton_kernel = patched
    print("[MI250_MOE_GEMV] patched triton_moe.invoke_fused_moe_wna16_triton_kernel", flush=True)


_warned = False


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
                except Exception as exc:
                    print(f"[MI250_MOE_GEMV] patch failed: {exc!r}", flush=True)

            spec.loader.exec_module = exec_module
            return spec
        return None


if _enabled():
    sys.meta_path.insert(0, _PatchFinder())
