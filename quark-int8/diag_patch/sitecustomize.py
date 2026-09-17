"""Diagnostic: log which checkpoint tensor name fails to resolve to a Parameter in
MergedColumnParallelLinear.load_weights (vLLM then passes the module itself as
`param` and dies on `param.data`). We log, then delegate to the original logic so
behaviour is otherwise unchanged."""
import importlib.abc
import os
import sys

TARGET = "vllm.model_executor.layers.linear"


def _enabled():
    return os.environ.get("VLLM_DIAG_LOADER", "0").lower() in ("1", "true")


def _apply(mod):
    cls = mod.MergedColumnParallelLinear
    if getattr(cls, "_diag_patched", False):
        return
    orig = cls.load_weights

    def load_weights(self, weights):
        weights = list(weights)
        for name, lw in weights:
            param = None
            try:
                if "." in name:
                    submodule, _, attr = name.rpartition(".")
                    param = getattr(self.get_submodule(submodule), attr, None)
                else:
                    param = getattr(self, name, None)
            except Exception as exc:  # submodule missing
                print(f"[diag-loader] LOOKUP-RAISED prefix={getattr(self,'prefix',None)!r} "
                      f"name={name!r} err={exc!r}", flush=True)
                continue
            if param is None:
                print(f"[diag-loader] UNRESOLVED prefix={getattr(self,'prefix',None)!r} "
                      f"name={name!r} shape={tuple(getattr(lw,'shape',()) or ())} "
                      f"dtype={getattr(lw,'dtype',None)} "
                      f"children={[n for n, _ in self.named_children()][:6]} "
                      f"params={[n for n, _ in self.named_parameters(recurse=False)][:6]}",
                      flush=True)
        yield from orig(self, weights)

    cls.load_weights = load_weights
    cls._diag_patched = True
    print("[diag-loader] installed", flush=True)


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

            def exec_module(m, _orig=orig_exec):
                _orig(m)
                try:
                    _apply(m)
                except Exception as exc:
                    print(f"[diag-loader] patch failed: {exc!r}", flush=True)

            spec.loader.exec_module = exec_module
            return spec
        return None


if _enabled():
    sys.meta_path.insert(0, _Finder())
