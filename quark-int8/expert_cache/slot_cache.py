"""Expert-cache prototype V1 -- *cost-model validation*, no memory re-layout yet.

What it does, per MoE forward:
  1. takes the routed expert ids (`topk_ids`) that vLLM already computed,
  2. classifies each unique expert as HOT (pinned in HBM) or COLD (conceptually
     living in DRAM) using a hot-list derived from measured routing frequency,
  3. for every COLD expert that is used, performs a *real* pinned-host -> device
     copy of that expert's weights (w13/w2/scales) back into its own rows. The
     payload is identical, so results are unchanged -- but the PCIe traffic is
     real, which is exactly what we need to measure.

Consequences: this V1 does NOT free any VRAM. It validates the two unknowns that
decide whether the full design (V2: slot-table shrinkage + LRU eviction) is worth
building:
  * the realised hit rate at a given offload percentage, and
  * the real per-layer latency cost of the misses.

V2 will then shrink the device tensors to `n_slots` rows, keep the cold experts in
pinned host memory, and remap `topk_ids` -> slot ids (the MoE kernel indexes its
weight tensor by topk_ids, so no kernel change is needed).

Env:
  EXPERT_CACHE_PCT        percent of experts treated as cold (default 12)
  EXPERT_CACHE_HOTLIST    hotlist.npz from build_hotlist.py
  EXPERT_CACHE_STATS      json file for counters (default /work/expert_cache_stats.json)
  EXPERT_CACHE_MIN_TOKENS only engage when tokens <= this (default 1000000 = always)
"""
from __future__ import annotations

import atexit
import json
import os
import threading
import time

import numpy as np
import torch


class ExpertOffloadSim:
    """Per-layer bookkeeping + real copies for the cold experts."""

    _registry: dict[str, "ExpertOffloadSim"] = {}
    _lock = threading.Lock()
    _global = {"layers": 0, "forwards": 0, "unique": 0, "cold_used": 0, "cold_miss": 0,
               "bytes_copied": 0, "copy_seconds": 0.0}

    def __init__(self, layer_name: str, w13, w2, s13, s2, cold_ids: np.ndarray):
        self.layer_name = layer_name
        self.cold_ids = cold_ids
        self.cold_mask = torch.zeros(512, dtype=torch.bool)
        self.cold_mask[cold_ids] = True
        self.cold_mask_dev = self.cold_mask.to(w13.device)

        self.w13, self.w2, self.s13, self.s2 = w13, w2, s13, s2
        # V1 only needs *real traffic* of the right size -> a single pinned scratch
        # buffer per layer (1.6 MB) instead of one pinned copy per cold expert
        # (that variant pinned ~46 GB across ranks and stalled the workers in D state).
        e0 = int(cold_ids[0])
        self.h13 = torch.empty_like(w13[e0], device="cpu", pin_memory=True)
        self.h2 = torch.empty_like(w2[e0], device="cpu", pin_memory=True)
        self.hs13 = torch.empty_like(s13[e0], device="cpu", pin_memory=True)
        self.hs2 = torch.empty_like(s2[e0], device="cpu", pin_memory=True)
        self.h13.copy_(w13[e0]); self.h2.copy_(w2[e0])
        self.hs13.copy_(s13[e0]); self.hs2.copy_(s2[e0])
        self.pos = None
        self.bytes_per_expert = (self.h13.numel() + self.h2.numel()) * 1 \
            + (self.hs13.numel() + self.hs2.numel()) * 4
        self.stats = {"forwards": 0, "unique": 0, "cold_used": 0, "misses": 0,
                      "bytes": 0, "copy_s": 0.0}
        with ExpertOffloadSim._lock:
            ExpertOffloadSim._registry[layer_name] = self
            ExpertOffloadSim._global["layers"] += 1

    # ------------------------------------------------------------------ runtime
    @torch.no_grad()
    def observe(self, topk_ids: torch.Tensor) -> None:
        """Count cold usage and pay the real copy cost (no id remapping in V1)."""
        try:
            flat = topk_ids.reshape(-1)
            uniq = torch.unique(flat)
            # vLLM uses -1 as the padding / "no expert" marker -> never a real expert
            uniq = uniq[(uniq >= 0) & (uniq < 512)]
            cold = uniq[self.cold_mask_dev[uniq]]
            n_uniq = int(uniq.numel())
            n_cold = int(cold.numel())
            st = self.stats
            st["forwards"] += 1
            st["unique"] += n_uniq
            st["cold_used"] += n_cold
            if n_cold:
                t0 = time.perf_counter()
                for e in cold.tolist():
                    self.w13[e].copy_(self.h13, non_blocking=True)
                    self.w2[e].copy_(self.h2, non_blocking=True)
                    self.s13[e].copy_(self.hs13, non_blocking=True)
                    self.s2[e].copy_(self.hs2, non_blocking=True)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                st["misses"] += n_cold
                st["bytes"] += n_cold * self.bytes_per_expert
                st["copy_s"] += dt
            g = ExpertOffloadSim._global
            g["forwards"] += 1
            g["unique"] += n_uniq
            g["cold_used"] += n_cold
            g["cold_miss"] += n_cold
            if g["forwards"] % 200 == 0:
                try:
                    ExpertOffloadSim.dump(os.environ.get(
                        "EXPERT_CACHE_STATS", "/work/expert_cache_stats.json"))
                except Exception:
                    pass
        except Exception as exc:  # stats must never take the engine down
            self.stats["errors"] = self.stats.get("errors", 0) + 1
            if self.stats["errors"] < 5:
                print(f"[expert-cache] observe error: {exc!r}", flush=True)

    # ------------------------------------------------------------------- report
    @classmethod
    def dump(cls, path: str) -> None:
        g = dict(cls._global)
        g["hit_rate_all_experts"] = (
            1.0 - g["cold_used"] / g["unique"] if g["unique"] else 1.0)
        g["layers"] = len(cls._registry)
        per_layer = {k: v.stats for k, v in list(cls._registry.items())[:8]}
        with open(path, "w") as f:
            json.dump({"global": g, "sample_layers": per_layer}, f, indent=2)
        print(f"[expert-cache] stats -> {path}\n[expert-cache] {json.dumps(g)}", flush=True)


def _safe_dump(path: str) -> None:
    try:
        ExpertOffloadSim.dump(path)
    except Exception:
        pass


def install(hotlist_path: str, offload_pct: int) -> None:
    from vllm.model_executor.layers.quantization.quark import quark_moe as qm

    z = np.load(hotlist_path)
    order = z["order"]                       # [layers, 512]
    n_cold = int(round(512 * offload_pct / 100))
    orig = qm.QuarkW8A8Int8MoEMethod.process_weights_after_loading

    def patched(self, layer):
        orig(self, layer)
        try:
            lid = getattr(layer, "layer_index", None)
            if lid is None:
                name = str(getattr(layer, "layer_name", ""))
                parts = [p for p in name.split(".")]
                lid = None
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                        lid = int(parts[i + 1]); break
            if lid is None or lid >= order.shape[0]:
                return
            cold = np.sort(order[lid][512 - n_cold:]).astype(np.int64)
            w13, w2 = layer.w13_weight, layer.w2_weight
            s13 = getattr(layer, "w13_weight_scale")
            s2 = getattr(layer, "w2_weight_scale")
            ExpertOffloadSim(str(getattr(layer, "layer_name", lid)),
                             w13.data, w2.data, s13.data, s2.data, cold)
        except Exception as exc:  # never break loading
            print(f"[expert-cache] install failed for layer: {exc!r}", flush=True)

    qm.QuarkW8A8Int8MoEMethod.process_weights_after_loading = patched

    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    orig_fwd = RoutedExperts.forward_modular

    def forward_modular(self, x, topk_weights, topk_ids, shared_experts=None,
                        shared_experts_input=None):
        sim = ExpertOffloadSim._registry.get(str(getattr(self, "layer_name", "")))
        if sim is not None:
            sim.observe(topk_ids)
        return orig_fwd(self, x, topk_weights, topk_ids, shared_experts,
                        shared_experts_input)

    RoutedExperts.forward_modular = forward_modular
    stats_path = os.environ.get("EXPERT_CACHE_STATS", "/work/expert_cache_stats.json")
    atexit.register(lambda: _safe_dump(stats_path))
    print(f"[expert-cache] installed (offload_pct={offload_pct}, cold/layer={n_cold}, "
          f"stats={stats_path})", flush=True)
