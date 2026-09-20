#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补丁自洽自检（2026-09-21）。

已可靠：② GEMV 四处副本哈希 ③ split-K 默认关闭 ④ QR C2/C3 marker。
**未收干净：① 树 vs base+补丁队列** —— 现用 patch -p1 整包应用会 rc=2（补丁含多文件 hunk，
目标文件没全部就位），因此那一节的 DRIFT 数字是假象。正确做法：把每个补丁触及的**所有**文件
都先按 base 就位，再**按文件粒度** `patch <file> < patch`，最后逐文件对拍。
参考：ops 文件的**手工**验证（把 0001+0005 只打在该文件上）是可靠的，结论是"与线上树仅差
_DCP_TOPK_CTX 那一处" ⇒ 该修复与 split-K 确实都没进队列。
"""
import argparse, hashlib, pathlib, re, shutil, subprocess, sys, tempfile

MAP = {
    "ops.py.orig": "v1/attention/ops/rocm_aiter_mla_sparse.py",
    "backend.py.orig": "v1/attention/backends/mla/rocm_aiter_mla_sparse.py",
    "indexer.py.orig": "model_executor/layers/sparse_attn_indexer.py",
}


def sha(p):
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.is_file() else "MISSING"


def reconstruct(tree, queue):
    base = queue / "base"
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="vp_"))
    for src, rel in MAP.items():
        f = base / src
        if f.is_file():
            d = tmp / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, d)
    log = []
    for p in sorted(queue.glob("*.patch")):
        r = subprocess.run(["patch", "-p1", "--batch", "--forward", "-i", str(p)],
                           cwd=tmp, capture_output=True, text=True)
        log.append((p.name, r.returncode))
    out = []
    for src, rel in MAP.items():
        rec, live = tmp / rel, tree / rel
        if not rec.is_file() or not live.is_file():
            out.append((rel, "SKIP", "缺文件")); continue
        a, b = rec.read_text(), live.read_text()
        if a == b:
            out.append((rel, "PASS", "逐字节一致"))
        else:
            al, bl = a.splitlines(), b.splitlines()
            dd = sum(1 for x, y in zip(al, bl) if x != y)
            out.append((rel, "DRIFT", "差异行约 %d，行数差 %d" % (dd, abs(len(al) - len(bl)))))
    shutil.rmtree(tmp, ignore_errors=True)
    return out, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="/home/qiba/ai/patches/gfx90a/ct_w4a16_dsv41_n0918/tree")
    ap.add_argument("--queue", default="quark-int8/dcp_patches")
    ap.add_argument("--repo", default="quark-int8/moe_gemv_patch")
    ap.add_argument("--ctr", default="hyperloom-local")
    a = ap.parse_args()
    tree, queue, repo = pathlib.Path(a.tree), pathlib.Path(a.queue), pathlib.Path(a.repo)
    bad = 0
    print("① base + 补丁队列 与线上树对拍")
    res, log = reconstruct(tree, queue)
    print("   应用: " + ", ".join("%s(rc=%d)" % (n, rc) for n, rc in log))
    for rel, st, note in res:
        print("   [%s] %-52s %s" % (st, rel, note))
        bad += (st == "DRIFT")
    print("② GEMV 四处副本哈希")
    for f in ("mi250_moe_gemv_gs.py", "mi250_moe_gemv_v2.py", "mi250_moe_gemv_v3.py", "sitecustomize.py"):
        t, r = sha(tree.parent / "moe_gemv" / f), sha(repo / f)   # moe_gemv 在补丁根目录，不在 tree/ 内
        ok = t == r and t != "MISSING"
        print("   [%s] %-24s tree=%s repo=%s" % ("PASS" if ok else "FAIL", f, t, r))
        bad += (not ok)
    print("③ split-K 默认关闭")
    ops = (tree / "v1/attention/ops/rocm_aiter_mla_sparse.py").read_text()
    m = re.search(r'MI250_SPARSE_SPLITK", "([^"]*)"', ops)
    val = m.group(1) if m else None
    ok = val == "0"
    print("   [%s] 默认值=%r（应为 0）" % ("PASS" if ok else "FAIL", val))
    bad += (not ok)
    for k in ("_DCP_TOPK_CTX", "_splitk_merge", "WRITE_LSE"):
        print("   [info] %-14s 出现 %d 次" % (k, ops.count(k)))
    print("④ QR C2/C3 marker")
    ps = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True)
    if a.ctr not in ps.stdout:
        print("   [SKIP] 容器 %s 未运行" % a.ctr)
    else:
        subprocess.run(["docker", "cp", "quark-int8/qr_marker_check.py", a.ctr + ":/tmp/qr_marker_check.py"],
                       capture_output=True)
        rr = subprocess.run(["docker", "exec", a.ctr, "python3", "/tmp/qr_marker_check.py"],
                            capture_output=True, text=True)
        ok = rr.stdout.strip() == "True True True"
        print("   [%s] 赋值行/gfx90/新条件 = %s" % ("PASS" if ok else "FAIL", rr.stdout.strip() or rr.stderr.strip()[:60]))
        bad += (not ok)
    print("⑤ 只挂载、未纳入补丁队列的文件（信息项）")
    cov = set(MAP.values())
    for p in sorted(tree.rglob("*.py")):
        rel = str(p.relative_to(tree))
        if rel not in cov and "moe_gemv" not in rel and "__pycache__" not in rel:
            print("   [info] %s" % rel)
    print()
    print("结果: " + ("全部通过" if bad == 0 else "%d 项失败" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())