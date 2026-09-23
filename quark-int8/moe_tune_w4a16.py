#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""int4_w4a16 MoE tile 实测调优器（gfx90a / MI250X）

纪律（都是本仓踩过的坑）：
  * **只把实测赢默认 ≥3% 的 M 桶写进表**；一个都不赢 ⇒ 不产出文件。
    （davetha 施工单 §实测：手搓种子表 58.8→-3%、276.4→-4%，劣于默认。）
  * 文件名一律由 vLLM 自己的 get_config_file_name() 生成 ⇒ 杜绝 device_name 手抄错。
    本机实测 device_name='AMD INSTINCT MI250 (MCM) OAM AC MBA'，
    而旧表写的是 'AMD_Instinct_MI250X_MI250' ⇒ 永远匹配不上（静默失效）。
  * M 命中方式是"最近邻桶"（configs[min(keys, key=|x-M|])）⇒ 桶要覆盖真实 M 分布。
  * 注入配置的方式是 monkey-patch try_get_optimal_moe_config：
    fused_experts_impl **没有** config= 形参（本仓 2026-09-20 实测核过签名）。
  * 默认基线走"生产同一路径"（block_shape=None），保证量的是运行时真正吃到的东西。

用法（空闲卡；被占时按硬门拒绝）:
  python3 moe_tune_w4a16.py --E 256 --N 256 --K 6144 --top-k 8 --group-size 32
"""
import argparse, json, os, subprocess, sys


def guard_busy():
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception as e:
        print("⚠ 读不到显存:", e); return
    used = [float(l.split()[-1]) / 2**30 for l in out.splitlines() if "Used" in l and l.strip()]
    busy = [u for u in used if u > 5.0]
    if busy and os.environ.get("ALLOW_BUSY", "0") != "1":
        sys.exit("❌ %d 个 GCD 占用 > 5 GiB ⇒ 别的会话在用，拒绝基准测试（ALLOW_BUSY=1 可强制）"
                 % len(busy))


def fname_of(E, N, dtype, block_shape):
    import importlib
    for mod in ("vllm.model_executor.layers.fused_moe.fused_moe",
                "vllm.model_executor.layers.fused_moe.config"):
        m = importlib.import_module(mod)
        f = getattr(m, "get_config_file_name", None)
        if f is not None:
            return f(E, N, dtype, block_shape)
    sys.exit("❌ 找不到 get_config_file_name —— 绝不手抄文件名")


def candidates():
    out = []
    for bm in (16, 32, 64, 128):
        for bn in (32, 64, 128):
            for bk in (32, 64, 128):
                for gm in (1, 4, 8, 16):
                    for nw in (4, 8):
                        out.append({"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn,
                                    "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": gm,
                                    "num_warps": nw, "num_stages": 3})
    return out


def bench(fn, iters=10, warmup=3):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(iters):
        st.record(); fn(); en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--N", type=int, default=256, help="= w2_shape[2]，分片后中间维")
    ap.add_argument("--K", type=int, default=6144, help="hidden_size")
    ap.add_argument("--top-k", dest="top_k", type=int, default=8)
    ap.add_argument("--group-size", dest="group_size", type=int, default=32)
    ap.add_argument("--M", default="1,2,4,8,16,32,64,128,256,512")
    ap.add_argument("--out-dir", default="/home/qiba/ai/config/moe-tuned")
    ap.add_argument("--max-cand", type=int, default=48)
    a = ap.parse_args()
    guard_busy()

    import torch
    import vllm.model_executor.layers.fused_moe.fused_moe as fm
    from vllm.triton_utils import triton

    dtype = "int4_w4a16"
    # 生产运行时的文件名以"无 block 后缀"为准（GLM 实跑 warning 即此形），同时把带 block
    # 的候选名也打印出来，便于对照日志确认哪个才被拾取。
    f_plain = fname_of(a.E, a.N, dtype, None)
    f_blk = fname_of(a.E, a.N, dtype, [1, a.group_size])
    print("[tune] 文件名（无 block，预期生效）:", f_plain)
    print("[tune] 文件名（带 block，备用）  :", f_blk)

    # 注入点：fused_experts_impl 无 config= ⇒ patch 配置查询函数
    _cur = {"cfg": None}
    _orig = fm.try_get_optimal_moe_config

    def _patched(w1_shape, w2_shape, top_k, dtt, M, block_shape=None):
        if _cur["cfg"] is not None:
            return dict(_cur["cfg"])
        return _orig(w1_shape, w2_shape, top_k, dtt, M, block_shape=block_shape)
    fm.try_get_optimal_moe_config = _patched

    dev = "cuda"
    torch.manual_seed(0)
    w1 = torch.randint(0, 256, (a.E, 2 * a.N, a.K // 2), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (a.E, a.K, a.N // 2), dtype=torch.uint8, device=dev)
    w1_s = (torch.rand((a.E, 2 * a.N, a.K // a.group_size), device=dev) * 0.01 + 0.001)
    w2_s = (torch.rand((a.E, a.K, a.N // a.group_size), device=dev) * 0.01 + 0.001)
    h0 = torch.randn(a.K, device=dev, dtype=torch.bfloat16)
    tw0 = torch.rand(a.top_k, device=dev, dtype=torch.float32)
    ti0 = torch.randint(0, a.E, (a.top_k,), device=dev, dtype=torch.int32)

    # ★ block_shape 必须与生产一致：GLM 运行期 warning 打印的文件名**不含 block 后缀**，
    #   而 get_moe_configs 里 block_shape = [b_n, b_k] if b_n and b_k else None
    #   ⇒ 生产查表时 block_shape 是 falsy。若这里固定传 [1,32]，量的就不是生产默认那条路。
    bs = {"v": None}

    def make(M):
        hs = h0.unsqueeze(0).expand(M, -1).contiguous()
        tw = tw0.unsqueeze(0).expand(M, -1).contiguous()
        ti = ti0.unsqueeze(0).expand(M, -1).contiguous()

        def fn():
            return fm.fused_experts_impl(
                hidden_states=hs, w1=w1, w2=w2, topk_weights=tw, topk_ids=ti,
                activation="silu", use_int4_w4a16=True,
                w1_scale=w1_s, w2_scale=w2_s, block_shape=bs["v"])
        return fn

    # 探一次：None 不行再退回 [1, group]，并明确打出用的是哪种（影响与生产的可比性）
    try:
        make(8)()
        torch.cuda.synchronize()
        print("[tune] block_shape=None 可跑（与生产查表路径一致）✓")
    except Exception as e:
        bs["v"] = [1, a.group_size]
        print("[tune] ⚠ block_shape=None 跑不通(%s) ⇒ 改用 %s；"
              % (str(e)[:80], bs["v"]))
        print("       此时**查表文件名会带 block 后缀**，需按带 block 的名字出表，见上面两个候选名。")

    cands = candidates()
    if len(cands) > a.max_cand:
        step = max(1, len(cands) // a.max_cand)
        cands = cands[::step][:a.max_cand]

    table = {}
    base_fail = 0
    for M in [int(x) for x in a.M.split(",")]:
        _cur["cfg"] = None                       # 基线 = 生产默认路径
        try:
            base_ms = bench(make(M))
        except Exception as e:
            print("  M=%-5d 基准构造失败: %r" % (M, e)); base_fail += 1; continue
        best, best_ms = None, base_ms
        for c in cands:
            _cur["cfg"] = c
            try:
                ms = bench(make(M), iters=6, warmup=1)
            except Exception:
                continue
            if ms < best_ms:
                best, best_ms = c, ms
        _cur["cfg"] = None
        gain = (base_ms / best_ms - 1) * 100 if best_ms > 0 else 0
        if best is not None and gain >= 3.0:
            table[str(M)] = best
            print("  M=%-5d 默认 %.3f → 最优 %.3f ms（+%.1f%%）✓ 入表" % (M, base_ms, best_ms, gain))
        else:
            print("  M=%-5d 默认 %.3f ms，最优只 +%.1f%% ⇒ 不入表" % (M, base_ms, gain))

    if not table:
        if base_fail:
            print("\n[tune] 结论：**基准构造失败 %d 个桶 ⇒ 本次没有产生任何有效测量**，" % base_fail)
            print('       因此不能得出「默认已够好」的结论（那是误读）；需先修基准的权重/scale 布局，')
            print("       使其与 CT WNA16 生产路径一致（int32 packed + bf16 scale），再重跑。")
        else:
            print("\n[tune] 结论：基准可跑，但没有任何桶能稳定赢默认 ≥3% ⇒ **不生成表**（默认已够好）。")
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    table["triton_version"] = triton.__version__     # vLLM 会 pop 掉它 ✓
    p = os.path.join(a.out_dir, f_plain)
    json.dump(table, open(p, "w"), indent=1)
    print("\n[tune] 已写出:", p)
    print("[tune] 生效判据：服务日志出现 'Using configuration from %s'，" % p)
    print("       而不再是 'Using default MoE config'。启动器需带 VLLM_TUNED_CONFIG_FOLDER。")
    return 0


if __name__ == "__main__":
    sys.exit(main())