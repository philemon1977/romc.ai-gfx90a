#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补丁自洽自检（2026-09-21）。

四节各自独立：① 树 vs base+补丁队列（**真门**）② GEMV 四处副本哈希 ③ split-K 默认关闭
④ QR C2/C3 marker。

① 的判据是**逐文件逐字节相等**：把 base/*.orig 铺到真实相对路径，按序打完队列里全部
补丁，再与线上挂载树对拍。任何 DRIFT 都意味着"有人在树上手工改过、而队列没记账"。

★ 关于"rc=2 假象"的更正（2026-09-21）：早前本节报 6 个补丁全 rc=2，并把它归因于
"多文件 hunk / 目标文件没就位"——**归因错了**。真实原因是 `patch -i <相对路径>` 把
-i 的路径按 **cwd（临时树）** 解析，于是"补丁文件本身找不到"⇒ rc=2，与补丁内容无关。
修法一行：`-i str(p.resolve())`。更正后 0001..0009 全部 rc=0，队列与树完全一致。
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
        # ★ 必须 resolve：patch -i 的相对路径按 cwd（临时树）解析，否则"补丁找不到"⇒ rc=2
        r = subprocess.run(["patch", "-p1", "--batch", "--forward", "-i", str(p.resolve())],
                           cwd=tmp, capture_output=True, text=True)
        log.append((p.name, r.returncode))
    # 覆盖面：队列里每个补丁触及的文件，是否都在 base 里有原件（否则那条改动无法重放）
    touched = set()
    for p in sorted(queue.glob("*.patch")):
        for line in p.read_text(errors="ignore").splitlines():
            if line.startswith("+++ b/"):
                touched.add(line[6:].strip())
    seeded = set(MAP.values())
    uncovered = sorted(touched - seeded)
    out = []
    for src, rel in MAP.items():
        rec, live = tmp / rel, tree / rel
        if not rec.is_file() or not live.is_file():
            out.append((rel, "SKIP", "缺文件")); continue
        a, b = rec.read_text(), live.read_text()
        if a == b:
            out.append((rel, "PASS", "逐字节一致"))
        else:
            import difflib
            d = [l for l in difflib.unified_diff(a.splitlines(), b.splitlines(), n=0)
                 if l[:1] in "+-" and not l.startswith(("+++", "---"))]
            out.append((rel, "DRIFT", "队列未记账的差异 %d 行（正=树有队列无，负=反之）" % len(d)))
    for rel in uncovered:
        out.append((rel, "NOBASE", "队列有补丁但 base 无原件 ⇒ 该文件无法重放校验"))
    shutil.rmtree(tmp, ignore_errors=True)
    return out, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="/home/qiba/ai/recipes/patches/gfx90a/ct_w4a16_dsv41_n0918/tree")
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