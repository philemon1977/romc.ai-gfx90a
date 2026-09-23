#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCP parity 判定：把新电池日志与已固化的 DCP=1 基线逐题比对。

用法: python3 compare_battery_vs_baseline.py [电池日志] [基线json]
基线来源: logs/parity_baseline_dcp1_1738.json（DCP=1 / TP8 / 32K / mmap，同一批固定 prompt）
判据（①）：固定 prompt 下 DCP=8 的答案应与 DCP=1 一致；不一致要看是"等价改写"还是"答错"。
"""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def latest_battery():
    fs = sorted(glob.glob(os.path.join(HERE, "logs", "glm_dcp_accept_*.log")), key=os.path.getmtime)
    return fs[-1] if fs else None


def parse_battery(path):
    fr, gs = [], []
    for ln in open(path, encoding="utf8", errors="ignore").read().splitlines():
        s = ln.strip()
        if s[:1] in ("'", '"') and "->" in s:
            fr.append(s)
        elif re.search(r"(OK|XX) gold=", s):
            gs.append(s)
    return fr, gs


def fact_pass(line):
    return "[PASS" in line


def main():
    bat = sys.argv[1] if len(sys.argv) > 1 else latest_battery()
    base_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "logs", "parity_baseline_dcp1_1738.json")
    if not bat or not os.path.exists(bat):
        print("找不到电池日志"); return
    base = json.load(open(base_path, encoding="utf8"))
    fr, gs = parse_battery(bat)
    print("电池日志: %s" % os.path.basename(bat))
    print("基线     : %s（%s）" % (os.path.basename(base_path), base.get("note", "")))
    print()
    print("== 事实召回（固定 prompt，逐题比对）==")
    n_ok = sum(1 for l in fr if fact_pass(l))
    print("  本轮 DCP 结果: %d/%d PASS" % (n_ok, len(fr)))
    for i, ln in enumerate(fr):
        b = base["fact_recall"][i] if i < len(base["fact_recall"]) else "(无基线)"
        same = ln.split("->")[-1].strip() == b.split("->")[-1].strip() if "->" in ln and "->" in b else False
        print("   %s 本轮: %s" % ("[同基线]" if same else "[不同  ]", ln[:96]))
        if not same:
            print("           基线: %s" % b[:96])
    print()
    print("== GSM8K（gold 命中）==")
    print("  本轮: %d OK / %d 题" % (sum(1 for l in gs if l.strip().startswith("OK")), len(gs)))
    print("  基线: %d OK / %d 题" % (sum(1 for l in base["gsm8k"] if l.strip().startswith("OK")), len(base["gsm8k"])))
    print()
    ok = n_ok == len(base["fact_recall"]) and all(fact_pass(l) for l in base["fact_recall"])
    print("⇒ ① parity %s（事实召回 %d/%d；GSM8K 见上）" % ("通过" if ok else "未通过", n_ok, len(fr)))
    if not ok:
        print("   不一致时的排查顺序：indexer 选择集（DSV41_IDX_DUMP）→ 分片过滤/长度（0003/0004）→ 合并（层 combine）")


if __name__ == "__main__":
    main()
