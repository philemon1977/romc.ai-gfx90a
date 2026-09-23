# -*- coding: utf-8 -*-
"""CT-int4 仓 FP4 专家 nibble 序修复（就地、无损、互逆）。

原理：CT int32 权重字内 nibble j（LSB 起）对应 K 的第 j 个元素（运行时
_unpack_gptq_int32_to_signed_int4 与我们的审计读取器一致）。转换器把源 FP4
字节读反了（hi 当偶数元素），于是每个专家行 = 源行「相邻元素对互换」。
互换每个字节内的两个 nibble，正好把相邻对换回来；组规模 32 元素 / 16 字节，
互换不跨字节 ⇒ weight_scale 逐组不变、仍然有效。

用法:
  python3 fix_fp4_nibble_order.py <repo> --measure
  python3 fix_fp4_nibble_order.py <repo> --apply [--shards N]
"""
import argparse, json, os, struct, sys, time
import numpy as np

def parse_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n

def collect(repo):
    idx = os.path.join(repo, "model.safetensors.index.json")
    wm = json.load(open(idx))["weight_map"]
    by_file = {}
    for k, fn in wm.items():
        if k.endswith(".weight_packed") and ".ffn.experts." in k:
            by_file.setdefault(fn, []).append(k)
    return by_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--shards", type=int, default=0, help="只处理前 N 个分片")
    ap.add_argument("--only", default="", help="只处理该分片文件名")
    ap.add_argument("--chunk", type=int, default=64 << 20)
    ap.add_argument("--force", action="store_true",
                    help="跳过分片占用检查与 journal 的半途检测（危险，需明确知道后果）")
    ap.add_argument("--journal", default="",
                    help="journal 路径（默认 <repo>/.nibble_fix_journal）")
    a = ap.parse_args()


    by_file = collect(a.repo)
    files = sorted(by_file)
    if a.only:
        files = [f for f in files if f == a.only]
        if not files:
            sys.exit("找不到分片 %s" % a.only)
    if a.shards:
        files = files[:a.shards]
    nkeys = sum(len(by_file[f]) for f in files)
    print("分片 %d 个 / 目标张量 %d 个" % (len(files), nkeys))

    # ── 安全门 1：仓库不能被别人正在使用（就地改写权重时读者会读到半状态） ──
    if a.apply and not a.force:
        exempt = set()
        _p = os.getpid()
        while _p and _p > 1:
            exempt.add(str(_p))
            try:
                with open("/proc/%d/stat" % _p) as fh:
                    _p = int(fh.read().rsplit(")", 1)[1].split()[1])
            except Exception:
                break
        hit = []
        for pid in os.listdir("/proc"):
            if pid in exempt:
                continue
            if not pid.isdigit():
                continue
            try:
                with open("/proc/%s/cmdline" % pid, "rb") as fh:
                    cl = fh.read().decode("utf-8", "ignore")
            except Exception:
                continue
            if a.repo.rstrip("/") in cl:
                hit.append(pid)
        if hit or os.popen("docker ps -q -f name=dsv41-ct-int4").read().strip():
            sys.exit("拒绝执行：仓库正被使用（pid=%s / 或本会话容器在运行）。"
                     "确认无读者后用 --force。" % ",".join(hit[:3]))

    # ── 安全门 2：journal —— 上次若在半途中断，禁止直接重跑 ──
    #    变换是自逆的：对"已修但 journal 未落 done"的分片重跑会把已修部分又翻回去
    jpath = a.journal or os.path.join(a.repo, ".nibble_fix_journal")
    pending = set()
    finished = set()
    if os.path.exists(jpath):
        for line in open(jpath, encoding="utf-8"):
            parts = line.strip().split("\t")
            if len(parts) == 2:
                fn, st = parts
                if st == "start":
                    pending.add(fn)
                else:
                    pending.discard(fn)
                    finished.add(fn)
    # ── 安全门 3：已修过的分片**必须拒绝**重复处理 ──
    #    变换是自逆的：再跑一次会把已修数据翻回坏状态（2026-09-20 合成仓实测确认）
    if finished and a.apply and not a.force:
        done_now = [f for f in files if f in finished]
        if done_now:
            sys.exit("拒绝执行：这些分片在 journal 里已标记 done（已修过）：%s…\n"
                     "再跑一次会**自逆回退**成坏状态。确认要回退请显式 --force。"
                     % ", ".join(sorted(done_now)[:3]))
    # ── 安全门 4：无 journal 的仓无法判断是否已修过（本仓两个真仓就是这种情况）──
    if a.apply and not os.path.exists(jpath) and not a.force:
        print("⚠️  本仓没有 journal：无法判断是否已被更早版本修过。\n"
              "    若已修过，本次运行会把它**翻回坏状态**。请先用：\n"
              "      python3 audit_fp4_nibble_order.py <源仓> <本仓>   # corr≈0.99 ⇒ 已修，别再跑\n"
              "    确认未修过再加 --force 继续。", flush=True)
        sys.exit(4)
    jfh = open(jpath, "a", encoding="utf-8") if a.apply else None
    if pending and a.apply and not a.force:
        sys.exit("拒绝执行：journal 显示上次在这些分片上中断（%s…）。"
                 "请先用 --only <分片> 逐个核对该分片是否已修（对比 audit_fp4_nibble_order.py 的 corr），"
                 "或明确 --force。" % ", ".join(sorted(pending)[:3]))

    total = 0
    t0 = time.time()
    for i, fn in enumerate(files, 1):
        p = os.path.join(a.repo, fn)
        hdr, base = parse_header(p)
        keys = by_file[fn]
        got = 0
        if jfh is not None:
            jfh.write("%s\tstart\n" % fn); jfh.flush()
        with open(p, "r+b") as f:
            for k in keys:
                b, e = hdr[k]["data_offsets"]
                if (e - b) % 4:
                    sys.exit("非 int32 对齐: %s %s" % (fn, k))
                off, end = base + b, base + e
                got += end - off
                if not a.apply:
                    continue
                while off < end:
                    n = min(a.chunk, end - off)
                    f.seek(off)
                    raw = f.read(n)
                    if len(raw) != n:      # ★ 短读必须报错：否则会把截短的缓冲写回，造成静默损坏
                        sys.exit("短读 %d/%d 于 %s@%d —— 中止（可能磁盘/文件异常）" % (len(raw), n, fn, off))
                    buf = np.frombuffer(raw, dtype=np.uint8).copy()
                    buf = ((buf >> 4) | (buf << 4)).astype(np.uint8)
                    f.seek(off)
                    f.write(buf.tobytes())
                    off += n
        total += got
        if jfh is not None:
            jfh.write("%s\tdone\n" % fn); jfh.flush()
        print("  [%2d/%2d] %-28s %-8s %7.2f GiB  %.1fs" % (
            i, len(files), fn, "已修" if a.apply else "计", got / 2**30, time.time() - t0), flush=True)
    print("合计 %.2f GiB, 用时 %.1f min, 模式=%s" % (total / 2**30, (time.time() - t0) / 60, "apply" if a.apply else "measure"))

main()
