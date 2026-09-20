#!/usr/bin/env python3
"""生成 dcp_patches 的 0007/0008/0009 三片（补上队列落后于线上树的部分）。

背景（2026-09-21）：队列冻结在 09-20 20:33，之后树上有三组改动没入队：
  A 缺陷① parity —— ops 用模块级 _DCP_TOPK_CTX + 后端 __init__ 写入（替掉"在 op 里
    就地调 get_current_vllm_config()"，那条路在 breakable-cudagraph 上下文里抛
    AssertionError: Current vLLM config is not set）；
  B DCP 取证开关 —— 后端 import os + 两处 MI250_DCP_DEBUG 打印（默认关）；
  C sparse split-K —— ops 里的 split-K 辅助函数与调用分支（MI250_SPARSE_SPLITK，默认 0）。

做法（不手改代码，全机械）：先把 base/*.orig 铺到真实相对路径并按序打 0001..0006
得到"重放中间态"，再对中间态与线上树做 diff，按 hunk **内容嗅探**把 hunk 分给三片；
未能归属的 hunk 一律硬失败（不许静默丢弃）。最后自证：重新重放 + 依次打 0007/0008/0009
必须与线上树逐文件 sha256 相等。

用法：
    python3 make_patch789.py --check-only     # 只重放对拍，不写片
    python3 make_patch789.py                  # 写片 + 自证
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "base"
TREE = Path("/home/qiba/ai/patches/gfx90a/ct_w4a16_dsv41_n0918/tree")

# 队列覆盖面：树内路径 -> base 里的上游原件
FILES = {
    "v1/attention/ops/rocm_aiter_mla_sparse.py": "ops.py.orig",
    "v1/attention/backends/mla/rocm_aiter_mla_sparse.py": "backend.py.orig",
    "model_executor/layers/sparse_attn_indexer.py": "indexer.py.orig",
}

EXISTING = ["0001_gfx90a_sparse_mla_lse.patch", "0002_gfx90a_indexer_dcp_topk.patch",
            "0003_gfx90a_attention_dcp.patch", "0004_gfx90a_dcp_local_seq_lens.patch",
            "0005_gfx90a_rocm_indexer_dcp_topk.patch", "0006_gfx90a_dcp_row_local_lengths.patch"]

# hunk 归属规则：(片号, 片名, 必须命中的特征)
RULES = [
    (7, "0007_gfx90a_dcp_topk_ctx.patch", ("_DCP_TOPK_CTX", "set_dcp_topk_ctx", "get_current_vllm_config")),
    (8, "0008_gfx90a_dcp_debug_switch.patch", ("MI250_DCP_DEBUG", "import os  # DCP 调试开关")),
    (9, "0009_gfx90a_sparse_splitk.patch", ("_SPARSE_SPLITK", "_splitk_", "split-K")),
]


def run(cmd, cwd=None, stdin=None):
    return subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True, text=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def build_replay(dst: Path, patches: list[str]) -> None:
    """把 base/*.orig 铺到真实相对路径，并按序打给定补丁。"""
    for rel, src in FILES.items():
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BASE / src, target)
    for name in patches:
        res = run(["patch", "-p1", "--batch", "--forward"], cwd=dst,
                  stdin=(HERE / name).read_text(encoding="utf-8"))
        if res.returncode != 0:
            raise SystemExit("replay failed at %s: rc=%s\n%s" % (name, res.returncode, res.stdout + res.stderr))


def split_hunks(diff_text: str) -> tuple[list[str], dict[str, list[str]]]:
    """把 unified diff 拆成 {文件: [hunk, ...]}，保留文件头顺序。"""
    files: dict[str, list[str]] = {}
    order: list[str] = []
    cur = None
    for line in diff_text.splitlines(keepends=True):
        if line.startswith("+++ "):
            cur = line[4:].strip()
            if cur.startswith("b/"):
                cur = cur[2:]
            files.setdefault(cur, [])
            order.append(cur)
        elif line.startswith("@@"):
            files[cur].append(line)
        elif cur and files[cur]:
            files[cur][-1] += line
    return order, files


def classify(hunk: str) -> int:
    hits = [n for n, _name, keys in RULES if any(k in hunk for k in keys)]
    if len(hits) != 1:
        raise SystemExit("hunk 归属不唯一/未知（命中 %s）：\n%s" % (hits, hunk[:400]))
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    for name in EXISTING:
        if not (HERE / name).is_file():
            raise SystemExit("缺少既有补丁：%s" % name)

    tmp = Path(tempfile.mkdtemp(prefix="replay789_"))
    build_replay(tmp, EXISTING)
    print("重放中间态：base + %s -> %s" % (", ".join(n[:4] for n in EXISTING), tmp))

    diff_text = ""
    for rel in FILES:
        res = run(["diff", "-u", str(tmp / rel), str(TREE / rel)])
        if res.returncode not in (0, 1):
            raise SystemExit("diff 失败 %s" % rel)
        if res.returncode == 0:
            continue
        body = res.stdout.split("\n", 2)[2]  # 去掉 diff 的两行时间戳头
        diff_text += "--- a/%s\n+++ b/%s\n%s" % (rel, rel, body)

    _order, files = split_hunks(diff_text)
    buckets: dict[int, dict[str, list[str]]] = {n: {} for n, _f, _k in RULES}
    for rel, hunks in files.items():
        for h in hunks:
            buckets[classify(h)].setdefault(rel, []).append(h)

    total = sum(len(v) for b in buckets.values() for v in b.values())
    print("hunk 分桶：%s（共 %d）" % ({n: sum(len(v) for v in buckets[n].values()) for n, _f, _k in RULES}, total))
    for n, fname, _k in RULES:
        print("  %s -> %s" % (fname, {k: len(v) for k, v in buckets[n].items()}))

    if args.check_only:
        print("check-only：未写片")
        return 0

    for n, fname, _k in RULES:
        out = []
        for rel in FILES:
            if rel in buckets[n]:
                out.append("--- a/%s\n+++ b/%s\n" % (rel, rel))
                out.extend(buckets[n][rel])
        (HERE / fname).write_text("".join(out), encoding="utf-8")
        print("wrote %s (%d bytes)" % (fname, (HERE / fname).stat().st_size))

    # 自证：全新重放 + 0007/0008/0009 必须与线上树逐文件 sha256 相等
    tmp2 = Path(tempfile.mkdtemp(prefix="verify789_"))
    build_replay(tmp2, EXISTING + [f for _n, f, _k in RULES])
    ok = True
    for rel in FILES:
        a, b = sha(tmp2 / rel), sha(TREE / rel)
        flag = "OK " if a == b else "DIFF"
        ok &= a == b
        print("  [%s] %-58s replay=%s tree=%s" % (flag, rel, a, b))
    if not ok:
        raise SystemExit("自证失败：重放结果与树不一致")
    print("自证通过：base + 0001..0009 == 线上树（逐文件 sha256）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
