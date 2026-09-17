"""Expert-cache V2 -- real slot table: frees VRAM by moving cold experts to DRAM.

Layout per MoE layer (per rank), all four weight tensors shrunk consistently:
    rows [0 .. H-1]              hot experts  (H = 512 - n_cold), compacted
    rows [H .. H+P-1]            P pool slots, LRU-managed cache for cold experts
so device memory drops by (n_cold - P) expert payloads per layer, and a cold
expert is streamed from pinned host memory only on a pool miss.

How it plugs into vLLM without kernel changes: the Triton INT8 MoE kernel indexes
its weight tensors by `topk_ids`, so we intercept `RoutedExperts.forward_modular`,
make the needed experts resident, and hand the kernel *slot* indices instead of
expert ids. Nothing else changes.

Overflow policy: a single forward may need more unique cold experts than P pool
slots (prefill chunks). vLLM's MoE is row-independent, so we split the token batch
into sub-batches whose union of cold experts fits in the pool and call the
original path once per sub-batch.

Cold experts live in *pageable* host memory plus one small pinned staging buffer
per layer: pinning tens of GB per rank previously stalled workers in D state.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time

import numpy as np
import torch


class SlotCache:
    _registry: dict[str, "SlotCache"] = {}
    _lock = threading.Lock()
    _global = {"layers": 0, "forwards": 0, "subbatches": 0, "unique": 0,
               "cold_used": 0, "pool_hit": 0, "miss": 0, "evict": 0,
               "bytes_freed": 0, "bytes_copied": 0, "copy_seconds": 0.0}

    def __init__(self, layer_name: str, layer, cold_ids: np.ndarray, pool_slots: int,
                 num_experts: int = 512):
        self.layer_name = layer_name
        self.num_experts = int(num_experts)
        self.n_cold = int(len(cold_ids))
        self.pool = int(pool_slots)
        self.hot_ids = np.setdiff1d(np.arange(self.num_experts, dtype=np.int64), cold_ids,
                                    assume_unique=False)
        self.H = int(len(self.hot_ids))
        self.S = self.H + self.pool

        w13, w2 = layer.w13_weight.data, layer.w2_weight.data
        s13, s2 = layer.w13_weight_scale.data, layer.w2_weight_scale.data
        dev = w13.device

        # ---- 1. park cold experts in pageable host memory -------------------
        cit = torch.as_tensor(cold_ids, dtype=torch.long)
        self.host13 = w13[cit].to("cpu", copy=True)
        self.host2 = w2[cit].to("cpu", copy=True)
        self.host_s13 = s13[cit].to("cpu", copy=True)
        self.host_s2 = s2[cit].to("cpu", copy=True)
        self.cold_pos = {int(e): i for i, e in enumerate(cold_ids)}

        # ---- 2. build shrunk device tensors --------------------------------
        new13 = torch.zeros((self.S, *w13.shape[1:]), dtype=w13.dtype, device=dev)
        new2 = torch.zeros((self.S, *w2.shape[1:]), dtype=w2.dtype, device=dev)
        ns13 = torch.zeros((self.S, *s13.shape[1:]), dtype=s13.dtype, device=dev)
        ns2 = torch.zeros((self.S, *s2.shape[1:]), dtype=s2.dtype, device=dev)
        hit = torch.as_tensor(self.hot_ids, dtype=torch.long)
        new13[: self.H].copy_(w13[hit]); new2[: self.H].copy_(w2[hit])
        ns13[: self.H].copy_(s13[hit]); ns2[: self.H].copy_(s2[hit])

        # ---- 3. mapping tables --------------------------------------------
        e2s = torch.full((self.num_experts,), -1, dtype=torch.int32, device=dev)
        e2s[hit.to(dev)] = torch.arange(self.H, dtype=torch.int32, device=dev)
        self.expert_to_slot = e2s                       # device, -1 = not resident
        self.slot_to_expert = [-1] * self.S             # host side, pool slots only
        self.lru: list[int] = []                        # pool slot order, MRU first

        # ---- 4. pinned staging (one expert payload) + swap in new tensors --
        e0 = int(self.hot_ids[0])
        self.st13 = torch.empty_like(w13[e0], device="cpu", pin_memory=True)
        self.st2 = torch.empty_like(w2[e0], device="cpu", pin_memory=True)
        self.st_s13 = torch.empty_like(s13[e0], device="cpu", pin_memory=True)
        self.st_s2 = torch.empty_like(s2[e0], device="cpu", pin_memory=True)
        self.w13, self.w2, self.s13, self.s2 = new13, new2, ns13, ns2
        self.bytes_per_expert = ((new13[0].numel() + new2[0].numel())
                                 + (ns13[0].numel() + ns2[0].numel()) * 4)
        self.freed = (self.n_cold - self.pool) * self.bytes_per_expert
        self.stats = {"forwards": 0, "unique": 0, "cold_used": 0, "pool_hit": 0,
                      "miss": 0, "evict": 0, "subbatches": 0, "copy_s": 0.0,
                      "bytes_copied": 0, "errors": 0}

        layer.w13_weight = torch.nn.Parameter(new13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(new2, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(ns13, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(ns2, requires_grad=False)
        del w13, w2, s13, s2, new13, new2, ns13, ns2
        torch.cuda.empty_cache()

        with SlotCache._lock:
            SlotCache._registry[layer_name] = self
            g = SlotCache._global
            g["layers"] += 1
            g["bytes_freed"] += self.freed
        print(f"[expert-cache-v2] {layer_name}: hot={self.H} pool={self.pool} "
              f"freed={self.freed/1e6:.1f}MB/rank", flush=True)

    # ---------------------------------------------------------------- helpers
    def _resident(self, e: int) -> bool:
        return int(self.expert_to_slot[e]) >= 0

    @torch.no_grad()
    def _fetch(self, expert: int) -> int:
        """Make `expert` resident, return its slot index."""
        slot = int(self.expert_to_slot[expert])
        if slot >= 0:
            self.stats["pool_hit"] += 1
            if slot in self.lru:
                self.lru.remove(slot)
            self.lru.insert(0, slot)
            return slot
        # miss -> need a slot
        if len(self.lru) >= self.pool:
            victim = self.lru.pop()
            ve = self.slot_to_expert[victim]
            if ve >= 0:
                self.expert_to_slot[ve] = -1
                self.slot_to_expert[victim] = -1
                self.stats["evict"] += 1
        else:
            victim = self.H + len(self.lru)
        self.lru.insert(0, victim)

        p = self.cold_pos[expert]
        self.st13.copy_(self.host13[p]); self.st2.copy_(self.host2[p])
        self.st_s13.copy_(self.host_s13[p]); self.st_s2.copy_(self.host_s2[p])
        self.w13[victim].copy_(self.st13, non_blocking=True)
        self.w2[victim].copy_(self.st2, non_blocking=True)
        self.s13[victim].copy_(self.st_s13, non_blocking=True)
        self.s2[victim].copy_(self.st_s2, non_blocking=True)
        self.expert_to_slot[expert] = victim
        self.slot_to_expert[victim] = expert
        self.stats["miss"] += 1
        self.stats["bytes_copied"] += self.bytes_per_expert
        return victim

    @torch.no_grad()
    def cold_need(self, topk_ids: torch.Tensor) -> torch.Tensor:
        flat = torch.unique(topk_ids.reshape(-1))
        flat = flat[(flat >= 0) & (flat < self.num_experts)]
        cold = flat[self.expert_to_slot[flat] < 0]
        return cold

    @torch.no_grad()
    def make_resident(self, topk_ids: torch.Tensor) -> None:
        """Ensure every routed expert is resident, then sync once."""
        cold = self.cold_need(topk_ids)
        n = int(cold.numel())
        st = self.stats
        st["forwards"] += 1
        st["unique"] += int(torch.unique(topk_ids.reshape(-1)).numel())
        st["cold_used"] += n
        if n == 0:
            return
        if n > self.pool:
            raise _PoolOverflow(n, self.pool)
        t0 = time.perf_counter()
        for e in cold.tolist():
            self._fetch(int(e))
        torch.cuda.synchronize()
        st["copy_s"] += time.perf_counter() - t0
        g = SlotCache._global
        g["forwards"] += 1
        g["cold_used"] += n
        g["miss"] += self.stats["miss"]
        # keep global counters close enough for reporting
        g["bytes_copied"] = sum(c.stats["bytes_copied"] for c in self._registry.values())

    @torch.no_grad()
    def remap(self, topk_ids: torch.Tensor) -> torch.Tensor:
        out = self.expert_to_slot[topk_ids.reshape(-1).to(torch.long)].to(topk_ids.dtype)
        return out.reshape(topk_ids.shape)

    # ----------------------------------------------------------------- report
    @classmethod
    def dump(cls, path: str) -> None:
        with cls._lock:
            g = dict(cls._global)
            layers = {k: v.stats for k, v in list(cls._registry.items())[:6]}
            if cls._registry:
                tot_cold = sum(c.stats["cold_used"] for c in cls._registry.values())
                tot_hit = sum(c.stats["pool_hit"] for c in cls._registry.values())
                g["pool_hit_rate"] = tot_hit / tot_cold if tot_cold else 1.0
                g["freed_GiB_total"] = g["bytes_freed"] / 1024**3
        with open(path, "w") as f:
            json.dump({"global": g, "sample_layers": layers}, f, indent=2)
        print(f"[expert-cache-v2] {json.dumps(g)}", flush=True)


class _PoolOverflow(Exception):
    def __init__(self, need: int, pool: int):
        super().__init__(f"need {need} cold experts > pool {pool}")
        self.need, self.pool = need, pool


def _safe_dump(path: str) -> None:
    try:
        SlotCache.dump(path)
    except Exception:
        pass


def install(hotlist_path: str, offload_pct: int, pool_slots: int) -> None:
    from vllm.model_executor.layers.quantization.quark import quark_moe as qm

    z = np.load(hotlist_path)
    order = z["order"]
    n_cold = int(round(512 * offload_pct / 100))
    orig = qm.QuarkW8A8Int8MoEMethod.process_weights_after_loading

    def patched(self, layer):
        orig(self, layer)
        try:
            n_exp = int(getattr(layer, "local_num_experts", 512))
            name = str(getattr(layer, "layer_name", ""))
            lid = None
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    lid = int(parts[i + 1]); break
            if lid is None or lid >= order.shape[0]:
                return
            top_k = int(getattr(layer, "top_k", 0) or
                        getattr(getattr(layer, "moe_config", None), "num_experts_per_tok", 0) or 1)
            pool = max(int(pool_slots), top_k)
            if n_exp == 512:
                cold = np.sort(order[lid][512 - n_cold:]).astype(np.int64)
            else:
                # synthetic ranking for small test models (tiny_int8)
                rng = np.random.default_rng(0)
                rank = rng.permutation(n_exp)
                cold = np.sort(rank[n_exp - max(1, round(n_exp * offload_pct / 100)):]).astype(np.int64)
            SlotCache(name or f"layer{lid}", layer, cold, pool, num_experts=n_exp)
        except Exception as exc:
            print(f"[expert-cache-v2] install failed: {exc!r}", flush=True)

    qm.QuarkW8A8Int8MoEMethod.process_weights_after_loading = patched

    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    orig_fwd = RoutedExperts.forward_modular

    def forward_modular(self, x, topk_weights, topk_ids, shared_experts=None,
                        shared_experts_input=None):
        cache = SlotCache._registry.get(str(getattr(self, "layer_name", "")))
        if cache is None:
            return orig_fwd(self, x, topk_weights, topk_ids, shared_experts,
                            shared_experts_input)
        need = cache.cold_need(topk_ids)
        if int(need.numel()) <= cache.pool:
            cache.make_resident(topk_ids)
            return orig_fwd(self, x, topk_weights, cache.remap(topk_ids),
                            shared_experts, shared_experts_input)

        # Too many distinct cold experts for one call -> process the batch in
        # sub-batches. Row-independent, so results are identical; the estimate is
        # verified and halved on overflow.
        rows = int(x.shape[0])
        n_sub = max(1, int(np.ceil(int(need.numel()) / max(cache.pool, 1))))
        step = max(1, rows // n_sub)
        rounds = 0
        while True:
            rounds += 1
            try:
                outs = []
                for s0 in range(0, rows, step):
                    sl = slice(s0, min(s0 + step, rows))
                    sub_ids = topk_ids[sl]
                    cache.make_resident(sub_ids)
                    outs.append(orig_fwd(self, x[sl], topk_weights[sl],
                                         cache.remap(sub_ids), shared_experts,
                                         shared_experts_input))
                break
            except _PoolOverflow:
                if step == 1:
                    raise
                step = max(1, step // 2)
        cache.stats["subbatches"] += rounds
        SlotCache._global["subbatches"] += rounds
        return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
        return orig_fwd(self, x, topk_weights, cache.remap(topk_ids),
                        shared_experts, shared_experts_input)

    RoutedExperts.forward_modular = forward_modular
    stats_path = os.environ.get("EXPERT_CACHE_STATS", "/work/expert_cache_stats.json")
    atexit.register(lambda: _safe_dump(stats_path))
    print(f"[expert-cache-v2] installed (offload={offload_pct}% -> {n_cold} cold/layer, "
          f"pool={pool_slots}, stats={stats_path})", flush=True)
