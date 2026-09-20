#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚合 worker 内 torch.profiler 的 JSON：decode 一步的 GPU 时间到底归谁。

产物来自 sitecustomize.py 的 MI250_PROF_WORKER 钩子（每个 TP rank 一个
worker_rank<N>.json，字段：skip/steps/kernels[{name,cuda_us,cpu_us,calls}]）。
关键判据：sum(各 kernel cuda_us)/steps 与探针实测的 step 墙钟时间对比 ——
差值就是"kernel 之外"的部分（启动/同步/CPU 调度/图重放空隙）。

用法：analyze_worker_prof.py PROF_DIR [top_n]
"""
import glob
import json
import os
import re
import sys

GROUPS = {
    "MoE(路由/专家/sort)": r"moe|expert|topk|sorting|grouped|align_block|mi250_moe_gemv|gemv",
    "注意力/MLA": r"attention|flash|paged|mla|reshape_and_cache|splitkv|merge_state|lse",
    "indexer(DSA 稀疏)": r"indexer|logits|ragged|dcp|topk_candidates|fetch_id",
    "W4A16 GEMM/反量化": r"wna16|awq|dequant|gptq|int4",
    "norm/激活": r"norm|silu|gelu|act",
    "通信": r"allreduce|all_gather|allgather|reduce_scatter|nccl|rccl|broadcast",
    "elementwise/copy/cast": r"elementwise|copy|cast|Cat|concat|permute|fill|zeros",
    "GEMM(其它)": r"gemm|Cijk|matmul|linear|bmm",
}


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "prof"
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 26
    files = sorted(glob.glob(os.path.join(d, "worker_rank*.json")))
    if not files:
        print("没找到 worker_rank*.json（目录：%s）" % d, "内容：", os.listdir(d)[:10] if os.path.isdir(d) else "目录不存在")
        return 1
    per_rank = {}
    for f in files:
        with open(f) as fh:
            per_rank[os.path.basename(f)] = json.load(fh)
    print("=== 参与聚合的 rank 文件：%d 个 ===" % len(per_rank))

    # 先看各 rank 的规模是否一致（TP 下应大致相同）
    for name, data in list(per_rank.items())[:8]:
        tot = sum(k["cuda_us"] for k in data["kernels"]) / 1000.0
        print("  %-22s steps=%d  内核种类=%3d  合计 CUDA %8.1f ms ⇒ %6.1f ms/步"
              % (name, data["steps"], len(data["kernels"]), tot, tot / max(1, data["steps"])))

    # 取 rank0 做代表（TP 各 rank 同构；若要跨 rank 均值可自行扩展）
    ref = per_rank[sorted(per_rank)[0]]
    steps = max(1, ref["steps"])
    agg = {}
    for k in ref["kernels"]:
        a = agg.setdefault(k["name"], [0.0, 0])
        a[0] += k["cuda_us"]
        a[1] += k["calls"]
    total_us = sum(v[0] for v in agg.values())
    print("\n== %s：%d 步窗口，合计 CUDA %.1f ms ⇒ **%.2f ms/步** ==" % (
        sorted(per_rank)[0], steps, total_us / 1000.0, total_us / 1000.0 / steps))
    print("\n%-62s %11s %8s %8s %10s" % ("kernel", "ms/步", "占比", "调用/步", "avg us"))
    for nm, (us, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top_n]:
        print("%-62s %11.3f %7.1f%% %8.1f %10.1f" % (
            nm[:62], us / 1000.0 / steps, 100 * us / total_us, n / steps, us / max(1, n)))
    print("\n--- 按类目归并（占 kernel 总时间）---")
    for label, pat in GROUPS.items():
        rx = re.compile(pat, re.I)
        us = sum(v[0] for nm, v in agg.items() if rx.search(nm))
        n = sum(v[1] for nm, v in agg.items() if rx.search(nm))
        print("%-24s %9.3f ms/步 %6.1f%%  %8.1f 次调用/步" % (
            label, us / 1000.0 / steps, 100 * us / total_us, n / steps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
