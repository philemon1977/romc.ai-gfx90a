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
    global _KCUR
    # 内核模块可选：mi250_moe_gemv_gs 是 group_size 泛化版（由 block_shape[1] 决定），
    # 默认用它以便同时支持 gs=128 (Ornith) 与 gs=32 (DSV4.1 CT-int4)。
    import importlib
    _modname = os.environ.get("MI250_MOE_GEMV_MODULE", "mi250_moe_gemv_gs")
    _m = importlib.import_module(_modname)
    global _MNAME
    _KCUR = _m
    _MNAME = _modname
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
            # 顺手暂存两个"专家是否分片"的关键事实，给 GEMM 级自检打印用：
            # expert_map=None ⇒ 本 rank 持全部专家（topk_ids 是全局号可直接索引 B）；
            # 否则 B 只含本 rank 的专家，topk_ids 必须先经 expert_map 本地化。
            try:
                _m._current["expert_map"] = kwargs.get("expert_map")
                _m._current["global_num_experts"] = kwargs.get("global_num_experts")
            except Exception:  # noqa: BLE001
                pass
            if os.environ.get("DSV41_ROUTE_DEBUG", "0") == "1" and not _routed.get("done"):
                _routed["done"] = True
                try:
                    import torch as _t
                    tw = kwargs.get("topk_weights", args[6] if len(args) > 6 else None)
                    print(f"[ROUTE] kwargs={sorted(kwargs.keys())} nargs={len(args)} "
                          f"shapes={[tuple(a.shape) if hasattr(a, 'shape') else type(a).__name__ for a in args[:9]]}",
                          flush=True)
                    if ids is None:
                        print("[ROUTE] topk_ids 取不到（args/kwargs 位置与假设不符，见上一行签名）", flush=True)
                    if tw is None:
                        print("[ROUTE] topk_weights 取不到", flush=True)
                    if ids is not None:
                        _i = ids.to(_t.int64)
                        print(f"[ROUTE] topk_ids{tuple(ids.shape)} uniq={int(_t.unique(_i).numel())} "
                              f"min={int(_i.min())} max={int(_i.max())}", flush=True)
                    if tw is not None:
                        _w = tw.float()
                        _s = _w.sum(-1)
                        print(f"[ROUTE] topk_weights{tuple(tw.shape)} sum[min={_s.min().item():.4f} "
                              f"max={_s.max().item():.4f}] w_max={_w.max().item():.4f} "
                              f"w_mean={_w.mean().item():.4f}", flush=True)
                except Exception as _e:  # noqa: BLE001
                    print(f"[ROUTE] failed: {_e!r}", flush=True)
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
        out = orig(A, B, C, B_scale, B_zp, topk_weights, sorted_token_ids, expert_ids,
                   num_tokens_post_padded, mul_routed_weight, top_k, config, compute_type,
                   use_int8_w8a16, use_int4_w4a16, block_shape)
        # ★ 真实请求上的自检（与我们的内核是否接管无关，只看 GEMM 级参数）：
        #   判据 = 暂存的 topk_ids 里存在 >=0 的真实专家号，且批足够小（真实请求）。
        if os.environ.get("MI250_MOE_GEMV_DEBUG", "0") == "1" and _refdone["n"] < 2:
            try:
                _ids = getattr(_m, "_current", {}).get("ids")
                if _ids is not None and _ids.numel() <= 512 and bool((_ids >= 0).any()):
                    _moe_selfcheck_real(A, B, C, B_scale, topk_weights, sorted_token_ids,
                                        expert_ids, num_tokens_post_padded, mul_routed_weight,
                                        top_k, block_shape, config)
            except Exception as _e:  # noqa: BLE001
                print(f"[MOECHK] 门控异常: {_e!r}", flush=True)
        return out

    module.invoke_fused_moe_wna16_triton_kernel = patched
    print("[MI250_MOE_GEMV] patched triton_moe.invoke_fused_moe_wna16_triton_kernel", flush=True)


_warned = False
_routed = {"done": False}
_refdone = {"n": 0}
_KCUR = None          # 内核模块的模块级引用（_apply 里的 _m 是局部变量）
_MNAME = None


def _moe_gemm_torch_ref(A, B, B_scale, topk_weights, sorted_token_ids, expert_ids,
                        mul_routed_weight, top_k, block_m, group, pairs, N, K):
    """独立 torch 参考：按 CT/**uint4b8** 口径显式反量化 B，再按 vLLM 的 sorted 布局算 GEMM。

    这是"上游 Triton WNA16 内核对不对"的裁判——我们自己的 GEMV 内核在本模型上
    因形状门（N%128、mul_routed_weight）根本不接管，所以 4-bit 专家反量化只可能
    是上游那个融合核在做，而它从未在 gfx90a 上被验证过。
    """
    import torch as _t
    with _t.no_grad():
        S = int(sorted_token_ids.numel())
        s = sorted_token_ids.to(_t.int64)
        epr = expert_ids.to(_t.int64)[_t.arange(S, device=s.device) // block_m]
        valid = (s >= 0) & (s < pairs)
        # 注意：上游核把结果写到 `sorted_token_ids[i]` 指向的行（= pair 下标 t*top_k+slot），
        # 所以参考也必须按 **pair 下标** 组织，而不是 sorted 位置。
        ref = _t.zeros(pairs, N, dtype=_t.float32, device=A.device)
        if not bool(valid.any()):
            return ref, s[valid], epr
        Af = A.float()
        # 两次 GEMM 的 A 布局不同：gemm1 的 A 是 hidden_states [M, K]（按 token 取），
        # gemm2 的 A 是 intermediate_cache2 [M*top_k, N]（已经按 pair 展开）。
        pair_indexed = (A.shape[0] == pairs)   # gemm2 的 A 已按 pair 展开
        tw = topk_weights.reshape(-1).float() if (mul_routed_weight and topk_weights is not None) else None
        for e in _t.unique(epr[valid]).tolist():
            rows = valid & (epr == e)
            be = B[e]                                   # [N, K//2] uint8，一字节两个 nibble
            lo = (be & 0x0F).to(_t.float32)
            hi = ((be >> 4) & 0x0F).to(_t.float32)
            codes = _t.empty(N, K, dtype=_t.float32, device=A.device)
            codes[:, 0::2] = lo - 8.0                   # uint4b8：真值 = (code-8)*scale
            codes[:, 1::2] = hi - 8.0
            W = codes * B_scale[e].float().repeat_interleave(group, dim=1)   # [N, K]
            tk = s[rows] if pair_indexed else (s[rows] // top_k)
            acc = Af[tk] @ W.t()
            if tw is not None:
                acc = acc * tw[s[rows]].unsqueeze(-1)
            ref[s[rows]] = acc
        return ref, s[valid], epr


def _moe_selfcheck_real(A, B, C, B_scale, topk_weights, sorted_token_ids, expert_ids,
                        num_tokens_post_padded, mul_routed_weight, top_k, block_shape,
                        config=None):
    """在**真实请求**的第一次 MoE 调用上：打印关键事实 + 与 torch 参考对拍 + 存盘。"""
    import torch as _t
    try:
        M, K = A.shape
        E, N, K2 = B.shape
        pairs = M * top_k
        group = int(block_shape[1]) if block_shape is not None else 32
        block_m = int(config["BLOCK_SIZE_M"]) if config is not None else 64
        _em = getattr(_KCUR, "_current", {}).get("expert_map")
        _ge = getattr(_KCUR, "_current", {}).get("global_num_experts")
        print(f"[MOECHK] A{tuple(A.shape)}{A.dtype} B{tuple(B.shape)}{B.dtype} "
              f"Bs{tuple(B_scale.shape)}{B_scale.dtype} pairs={pairs} top_k={top_k} "
              f"gs={group} mul_w={mul_routed_weight} global_experts={_ge} "
              f"expert_map={'None' if _em is None else tuple(_em.shape)}", flush=True)
        ei = expert_ids.to(_t.int64)
        print(f"[MOECHK] expert_ids: min={int(ei.min())} max={int(ei.max())} "
              f"uniq={int(_t.unique(ei).numel())} numel={ei.numel()} "
              f"sorted numel={sorted_token_ids.numel()} num_pad={num_tokens_post_padded}", flush=True)
        if topk_weights is not None:
            _w = topk_weights.float()
            print(f"[MOECHK] topk_weights sum[min={_w.sum(-1).min():.4f} max={_w.sum(-1).max():.4f}]", flush=True)
        ref, real_idx, epr = _moe_gemm_torch_ref(
            A, B, B_scale, topk_weights, sorted_token_ids, expert_ids,
            mul_routed_weight, top_k, block_m, group, pairs, N, K)
        n_real = int(real_idx.numel())
        if n_real == 0:
            print("[MOECHK] 本批无真实对（全 padding），跳过数值对拍", flush=True)
            return
        got = C.view(-1, N)[:pairs].float()[real_idx]
        a, b = got, ref[real_idx]
        rel = (a - b).abs().mean().item() / (b.abs().mean().item() + 1e-9) * 100
        print(f"[MOECHK] ★真实对 {n_real} 个（覆盖专家 {int(_t.unique(epr[ (sorted_token_ids.to(_t.int64)>=0) & (sorted_token_ids.to(_t.int64)<pairs) ]).numel())} 个, "
              f"mul_w={mul_routed_weight}）: 上游核 vs torch参考 相对误差 = {rel:.3f}%  "
              f"|ref|均值={b.abs().mean():.4f} maxabs={(a-b).abs().max():.4e}", flush=True)
        if _refdone["n"] == 0:
            try:
                _t.save({"A": A[:16].clone(), "B": B.clone(), "B_scale": B_scale.clone(),
                         "C": C.clone(), "ref": ref, "sorted": sorted_token_ids.clone(),
                         "expert_ids": expert_ids.clone(), "real_idx": real_idx,
                         "topk_weights":
                         (topk_weights.clone() if topk_weights is not None else None),
                         "top_k": top_k, "group": group, "pairs": pairs,
                         "N": N, "K": K, "mul_routed_weight": mul_routed_weight,
                         "block_m": block_m},
                        "/tmp/moe_real_capture.pt")
                print("[MOECHK] 已存 /tmp/moe_real_capture.pt（可离线复算）", flush=True)
            except Exception as _e:  # noqa: BLE001
                print(f"[MOECHK] 存盘失败: {_e!r}", flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[MOECHK] 自检异常: {_e!r}", flush=True)
    finally:
        _refdone["n"] += 1


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


# ============================================================================
# worker 内 torch.profiler（2026-09-21）
# 为什么需要：vLLM v1 把模型跑在**独立 worker 进程**里，driver 进程的 torch.profiler
# 只能看到 8.8 us 的 hipDeviceSynchronize（实测）。rocprofv3 又在 RCCL 初始化处死锁，
# 所以唯一的定位手段是**在 worker 进程内**挂钩。默认关闭（MI250_PROF_WORKER 未设时零影响）。
#
# 用法（在一次性容器里，env 由 docker run -e 传入）：
#   MI250_PROF_WORKER=1 MI250_PROF_SKIP=10 MI250_PROF_STEPS=8 MI250_PROF_OUT=/work/prof
#   ⇒ 抓第 10..17 次 execute_model，各 rank 写 worker_rank<N>.txt
# ============================================================================
import time as _time  # noqa: E402

PROF_TARGET = "vllm.v1.worker.gpu_model_runner"
_PROF_ON = os.environ.get("MI250_PROF_WORKER", "0").strip().lower() in _ENV_ON
_prof = {"n": 0, "p": None, "skip": 0, "steps": 0}


def _prof_step(self, orig, args, kwargs):
    _prof["n"] += 1
    n = _prof["n"]
    if n == _prof["skip"]:
        from torch.profiler import ProfilerActivity, profile
        _prof["p"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        _prof["p"].__enter__()
        print(f"[MI250_PROF] 开始抓：execute_model #{n}，之后 {_prof['steps']} 步", flush=True)
    t0 = _time.perf_counter()
    out = orig(self, *args, **kwargs)
    dt = _time.perf_counter() - t0
    if _prof["p"] is not None:
        print(f"[MI250_PROF] step#{n} {dt * 1000:.1f} ms", flush=True)
    if _prof["p"] is not None and n == _prof["skip"] + _prof["steps"] - 1:
        try:
            import torch
            _prof["p"].__exit__(None, None, None)
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            outdir = os.environ.get("MI250_PROF_OUT", "/work/prof")
            os.makedirs(outdir, exist_ok=True)
            path = os.path.join(outdir, "worker_rank%d.txt" % rank)
            ka = _prof["p"].key_averages()
            with open(path, "w") as fh:
                fh.write("# execute_model #%d..#%d (共 %d 步)\n"
                         % (_prof["skip"], _prof["skip"] + _prof["steps"] - 1, _prof["steps"]))
                fh.write(ka.table(sort_by="cuda_time_total", row_limit=60))
            # 文本表会截断 kernel 名 ⇒ 同时写 JSON（精确可聚合）
            import json as _json
            with open(os.path.join(outdir, "worker_rank%d.json" % rank), "w") as jh:
                _json.dump({"skip": _prof["skip"], "steps": _prof["steps"],
                            "kernels": [{"name": e.key,
                                         "cuda_us": getattr(e, "self_device_time_total", None) or getattr(e, "cuda_time_total", 0.0),
                                         "cpu_us": getattr(e, "self_cpu_time_total", None) or getattr(e, "cpu_time_total", 0.0),
                                         "calls": e.count}
                                        for e in ka]}, jh)
            print(f"[MI250_PROF] rank{rank} 已写 {path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[MI250_PROF] 写表失败: {exc!r}", flush=True)
        finally:
            _prof["p"] = None
    return out


def _apply_prof(module):
    cls = getattr(module, "GPUModelRunner", None)
    if cls is None or getattr(cls, "_mi250_prof", False):
        return
    _prof["skip"] = int(os.environ.get("MI250_PROF_SKIP", "10"))
    _prof["steps"] = int(os.environ.get("MI250_PROF_STEPS", "8"))
    orig = cls.execute_model

    def execute_model(self, *a, **kw):
        return _prof_step(self, orig, a, kw)

    cls.execute_model = execute_model
    cls._mi250_prof = True
    print(f"[MI250_PROF] 已在 worker 内挂钩 execute_model（skip={_prof['skip']} "
          f"steps={_prof['steps']}）", flush=True)


class _ProfFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != PROF_TARGET:
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
                    _apply_prof(mod)
                except Exception as exc:  # noqa: BLE001
                    print(f"[MI250_PROF] patch failed: {exc!r}", flush=True)

            spec.loader.exec_module = exec_module
            return spec
        return None


if _enabled():
    sys.meta_path.insert(0, _PatchFinder())
if _PROF_ON:
    sys.meta_path.insert(0, _ProfFinder())

