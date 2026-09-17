"""Lazy installer for the expert-cache prototypes.

EXPERT_CACHE_MODE=sim  -> V1: no memory re-layout, real copies only (cost model)
EXPERT_CACHE_MODE=slot -> V2: shrunk slot table + LRU + real VRAM freed

Kept as an import hook so importing this file never imports vLLM itself
(an early vLLM import silently disables vLLM's INFO logging).

Env:
  EXPERT_CACHE=1
  EXPERT_CACHE_MODE=slot|sim
  EXPERT_CACHE_HOTLIST=/ec/hotlist.npz
  EXPERT_CACHE_PCT=8
  EXPERT_CACHE_POOL=8          (V2 only: pool slots per layer)
  EXPERT_CACHE_STATS=/work/expert_cache_stats.json
"""
import importlib.abc
import os
import sys

TARGET = "vllm.model_executor.layers.quantization.quark.quark_moe"


def _enabled() -> bool:
    return os.environ.get("EXPERT_CACHE", "0").lower() in ("1", "true", "yes")


def _apply(module) -> None:  # noqa: ARG001
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    hotlist = os.environ.get("EXPERT_CACHE_HOTLIST", "/ec/hotlist.npz")
    pct = int(os.environ.get("EXPERT_CACHE_PCT", "8"))
    mode = os.environ.get("EXPERT_CACHE_MODE", "sim").lower()

    if mode == "slot":
        import slot_cache_v2  # noqa: PLC0415

        slot_cache_v2.install(hotlist, pct, int(os.environ.get("EXPERT_CACHE_POOL", "8")))
    else:
        import slot_cache  # noqa: PLC0415

        slot_cache.install(hotlist, pct)


class _Finder(importlib.abc.MetaPathFinder):
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
                    print(f"[expert-cache] install error: {exc!r}", flush=True)

            spec.loader.exec_module = exec_module
            return spec
        return None


if _enabled():
    sys.meta_path.insert(0, _Finder())
