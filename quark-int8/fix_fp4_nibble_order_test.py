# -*- coding: utf-8 -*-
"""fix_fp4_nibble_order.py 的守门测试（合成小仓，不碰真仓）。

覆盖四道安全门 + 一次正向验证：
  1) 首次 apply 变换正确、journal 落 start/done
  2) 已修过的分片再 apply（不带 --force）⇒ **必须拒绝**（自逆会损坏）
  3) journal 留 start 无 done（半途中断）⇒ 必须拒绝
  4) 完全没有 journal（老仓）⇒ 警告 + 退出码 4（防止对已修仓误跑）
  5) 显式 --force ⇒ 允许（并说明会回退）
用法: python3 fix_fp4_nibble_order_test.py [工具路径]
"""
import json, os, subprocess, sys
import numpy as np
from safetensors.numpy import save_file, load_file

REPO = "/tmp/mrepo"                     # ★ 不要用会与测试脚本名互为前缀的路径（会触发占用误报）
TOOL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "fix_fp4_nibble_order.py")
SHARD = "model-00001-of-00001.safetensors"
KEY = "layers.0.ffn.experts.0.w1.weight_packed"
JOURNAL = os.path.join(REPO, ".nibble_fix_journal")

def build():
    os.makedirs(REPO, exist_ok=True)
    for f in os.listdir(REPO):
        os.remove(os.path.join(REPO, f))
    rng = np.random.default_rng(0)
    wp = rng.integers(-(2**31), 2**31 - 1, size=(8, 4), dtype=np.int32)
    save_file({KEY: wp, KEY.replace("weight_packed", "weight_scale"): np.ones((8, 1), dtype=np.uint8)},
              os.path.join(REPO, SHARD))
    json.dump({"weight_map": {KEY: SHARD, KEY.replace("weight_packed", "weight_scale"): SHARD}},
              open(os.path.join(REPO, "model.safetensors.index.json"), "w"))
    return wp.copy()

def run(*args):
    return subprocess.run([sys.executable, TOOL, REPO] + list(args), capture_output=True, text=True)

def packed():
    return load_file(os.path.join(REPO, SHARD))[KEY]

def nib_swap(a):
    u = a.view(np.uint32); m = np.uint32(0x0F0F0F0F)
    return (((u & m) << np.uint32(4)) | ((u >> np.uint32(4)) & m)).astype(np.uint32).view(np.int32)

def last(r):
    txt = (r.stdout or r.stderr).strip().splitlines()
    return txt[-1][:150] if txt else "(无输出)"

fails = 0
def check(cond, name, extra=""):
    global fails
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, extra))
    fails += 0 if cond else 1

orig = build()
print("=== 1) 首次 apply --force（合成仓在 /tmp，无真实读者） ===")
r = run("--apply", "--force"); print("     ", last(r))
check(r.returncode == 0 and np.array_equal(packed(), nib_swap(orig)), "变换正确（字节内 nibble 互换）")
check(os.path.exists(JOURNAL) and "done" in open(JOURNAL).read(), "journal 记录了 start/done")

print("=== 2) 已修过的分片再 apply（不带 --force）⇒ 必须拒绝 ===")
r = run("--apply"); print("     ", last(r))
check(r.returncode != 0 and np.array_equal(packed(), nib_swap(orig)), "拒绝重复处理且数据未被改动")

print("=== 3) 半途中断（journal 只有 start）⇒ 必须拒绝 ===")
open(JOURNAL, "w").write(SHARD + "\tstart\n")
r = run("--apply"); print("     ", last(r))
check(r.returncode != 0, "检出半途状态并拒绝")

print("=== 4) 完全没有 journal（老仓）⇒ 警告 + 退出码 4 ===")
os.remove(JOURNAL)
r = run("--apply"); print("     ", last(r))
check(r.returncode == 4 and "journal" in (r.stdout + r.stderr), "无 journal 时拒绝并要求先 audit")

print("=== 5) 显式 --force ⇒ 允许（自逆回退为已知行为） ===")
r = run("--apply", "--force"); print("     ", last(r))
check(r.returncode == 0 and np.array_equal(packed(), orig), "第二次 apply 确实自逆回退（故必须挡住）")

print("=== 结论：%s ===" % ("全部通过" if not fails else "%d 项失败" % fails))
sys.exit(1 if fails else 0)
