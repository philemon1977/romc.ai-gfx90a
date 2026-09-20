#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 MI250X runner 落到 Magpie（第三方代码，故同样用幂等 applier）。

**必须在容器内运行**（Magpie 装在容器的 dist-packages，宿主机看不到）：
    docker exec hyperloom-local python3 /home/qiba/ROCm.AI/hyperloom/patches-local/apply_mi250x_runner.py

做三件事（都可重复执行）：
  1) 把工作区的 hyperloom/patches-local/magpie-scripts/vllm_mi250x.sh
     拷进 <MAGPIE_PATH>/Magpie/scripts/benchmark/（内容不一致才覆盖）；
  2) 把 "vllm_mi250x.sh" 加进 Magpie/modes/benchmark/benchmarker.py 的
     MAGPIE_BUILTIN_SCRIPTS（该 frozenset 是"Magpie 自带、支持 server/client 拆分"的脚本清单）；
  3) 给 Magpie/modes/benchmark/image_selector.py 的 arch->runner 表加 "gfx90a": "mi250x"
     （自探测路径；Hyperloom 显式传 runner_type 时不走这里，但补上才叫一致）。
"""
import os
import pathlib
import shutil
import sys

MAGPIE_PATH = os.environ.get("MAGPIE_PATH", "").strip() or "/usr/local/lib/python3.12/dist-packages"
SRC = pathlib.Path("/home/qiba/ROCm.AI/hyperloom/patches-local/magpie-scripts/vllm_mi250x.sh")
DEST_DIR = pathlib.Path(MAGPIE_PATH) / "Magpie" / "scripts" / "benchmark"
DEST = DEST_DIR / "vllm_mi250x.sh"
BENCHMARKER = pathlib.Path(MAGPIE_PATH) / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
SELECTOR = pathlib.Path(MAGPIE_PATH) / "Magpie" / "modes" / "benchmark" / "image_selector.py"


def main():
    if not SRC.is_file():
        sys.exit("源脚本不在: " + str(SRC))
    if not DEST_DIR.is_dir():
        sys.exit("Magpie scripts 目录不在: " + str(DEST_DIR))

    # 1) 拷脚本
    new = SRC.read_text()
    old = DEST.read_text() if DEST.is_file() else None
    if old == new:
        print("  脚本        already")
    else:
        shutil.copyfile(SRC, DEST)
        os.chmod(DEST, 0o755)
        print("  脚本        " + ("updated" if old is not None else "created"))

    # 2) 注册进 MAGPIE_BUILTIN_SCRIPTS
    text = BENCHMARKER.read_text()
    if '"vllm_mi250x.sh"' in text:
        print("  BUILTIN     already")
    elif 'MAGPIE_BUILTIN_SCRIPTS' in text and '        "vllm_mi300x.sh",' in text:
        text = text.replace('        "vllm_mi300x.sh",',
                            '        "vllm_mi250x.sh",\n        "vllm_mi300x.sh",', 1)
        BENCHMARKER.write_text(text)
        print("  BUILTIN     applied")
    else:
        print("  BUILTIN     MISSING ANCHOR")

    # 3) Magpie 自己的 arch -> runner 映射
    text = SELECTOR.read_text()
    if '"gfx90a"' in text:
        print("  arch 映射   already")
    elif '            "gfx942": "mi300x",' in text:
        text = text.replace('            "gfx942": "mi300x",',
                            '            "gfx90a": "mi250x",   # MI250X (gfx90a, 每 GCD 一设备)\n            "gfx942": "mi300x",', 1)
        SELECTOR.write_text(text)
        print("  arch 映射   applied")
    else:
        print("  arch 映射   MISSING ANCHOR")

    import ast
    for p in (BENCHMARKER, SELECTOR):
        ast.parse(p.read_text())
    print("  语法        两份 .py 均 AST_OK")
    print("  校验        脚本存在=" + str(DEST.is_file())
          + " | BUILTIN 含 mi250x=" + str('"vllm_mi250x.sh"' in BENCHMARKER.read_text())
          + " | selector 含 gfx90a=" + str('"gfx90a"' in SELECTOR.read_text()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
