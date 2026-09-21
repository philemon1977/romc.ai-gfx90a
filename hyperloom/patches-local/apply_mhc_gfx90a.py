#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''幂等 applier：把 gfx90a 排除出 TileLang MHC（vLLM model_executor/layers/mhc.py）。

为什么需要：上游 _has_tilelang_mhc() 只排 gfx942（注释自己写着 TileLang MHC
produces incorrect results on gfx942）。gfx90a 是 wave64，TileLang 的 mHC 融合核会把
sync 从 if 内提到 if 外（自报 [ThreadSync] ... tx < 32），非确定算错 layer_input。
凡 config 带 hc_mult / hc_sinkhorn_iters 的模型（glm5next / deepseek_v4 / deepseek_v41）
在本机都必须走 torch/triton 回落，否则整模型输出是「自信的乱码」。

用法：
  ./apply_mhc_gfx90a.py --check      # 只看状态（不写）
  ./apply_mhc_gfx90a.py --dry-run    # 临时副本上试打 + py_compile（不碰原树）
  ./apply_mhc_gfx90a.py --apply      # 打（幂等）
  ./apply_mhc_gfx90a.py --revert     # 回退成上游原文
  --target <file> 指定文件（默认 src/vllm-master 那棵 editable 树）
'''
import argparse, os, pathlib, py_compile, shutil, sys, tempfile

AI = pathlib.Path('/home/qiba/ai')
DEFAULT_TARGET = AI / 'src/vllm-master/vllm/model_executor/layers/mhc.py'
MARKER = 'gfx90a-host patch: tilelang MHC (glm5next-int8 port)'

UPSTREAM = (
    '    if current_platform.is_rocm():\n'
    '        from vllm.platforms.rocm import on_gfx942\n'
    '\n'
    '        # TileLang MHC currently produces incorrect results on gfx942. Keep\n'
    '        # gfx942 on the existing torch/triton fallbacks until that path is fixed.\n'
    '        return not on_gfx942()\n'
)

PATCHED = (
    '    if current_platform.is_rocm():\n'
    '        from vllm.platforms.rocm import on_gfx90a, on_gfx942\n'
    '\n'
    '        # TileLang MHC currently produces incorrect results on gfx942. Keep\n'
    '        # gfx942 on the existing torch/triton fallbacks until that path is fixed.\n'
    '        #\n'
    '        # --8<-- ' + MARKER + ' --8<--\n'
    '        # 同 patches/gfx90a/ct_w4a16_dsv41/mhc.py（补丁组 12）：gfx90a 是 wave64，\n'
    '        # TileLang 自报 [ThreadSync] Hoisting sync ... tx < 32，mhc_pre_delayed_tilelang\n'
    '        # 非确定算错 layer_input（同输入 4 次 maxabs 1.85/9.8e-4/2.17/1.97，量级约 1.2）。\n'
    '        if on_gfx90a():\n'
    '            return False\n'
    '        return not on_gfx942()\n'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default=str(DEFAULT_TARGET))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true')
    g.add_argument('--apply', action='store_true')
    g.add_argument('--revert', action='store_true')
    g.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    t = pathlib.Path(a.target)
    if not t.is_file():
        print('MISSING ' + str(t))
        return 2
    src = t.read_text()
    marked = MARKER in src

    if a.check:
        if marked:
            print('PATCHED  (marker 在位) ' + str(t))
            return 0
        if UPSTREAM in src:
            print('UNPATCHED (上游原文在位) ' + str(t))
            print('   => HAS_TILELANG_MHC 在 gfx90a 上仍为 True')
            return 1
        print('UNKNOWN  既无 marker 也不匹配上游原文，需人工看: ' + str(t))
        return 3

    if a.apply and marked:
        print('已经是补丁版，no-op')
        return 0
    if a.revert and not marked:
        print('本来就是上游原文，no-op')
        return 0

    n_up = src.count(UPSTREAM)
    n_pa = src.count(PATCHED)
    if a.apply and n_up != 1:
        print('上游块匹配数 = %d（要恰好 1）=> 不敢动，人工核对' % n_up)
        return 3
    if a.revert and n_pa != 1:
        print('补丁块匹配数 = %d（要恰好 1）=> 不敢动，人工核对' % n_pa)
        return 3

    new = src.replace(UPSTREAM, PATCHED) if a.apply or a.dry_run else src.replace(PATCHED, UPSTREAM)
    if new == src:
        print('替换后无变化，放弃')
        return 3

    if a.dry_run:
        d = tempfile.mkdtemp()
        cp = os.path.join(d, 'mhc.py')
        open(cp, 'w').write(new)
        py_compile.compile(cp, doraise=True)
        shutil.rmtree(d, ignore_errors=True)
        print('dry-run 通过：可干净替换且 py_compile 通过（原树未被触碰）')
        print('  行数 %d -> %d (+%d)' % (len(src.splitlines()), len(new.splitlines()),
                                        len(new.splitlines()) - len(src.splitlines())))
        return 0

    bak = str(t) + '.bak_mhc_gfx90a'
    if not os.path.exists(bak):
        shutil.copy2(t, bak)
        print('  备份 -> ' + bak)
    tmp = str(t) + '.new'
    open(tmp, 'w').write(new)
    os.replace(tmp, t)
    py_compile.compile(str(t), doraise=True)
    print('已' + ('打补丁' if a.apply else '回退为上游原文') + '：' + str(t))
    print('  纯 CPU 验收: python -c "from vllm.model_executor.layers.mhc import HAS_TILELANG_MHC as H; print(H)"  => 应为 False')
    return 0


if __name__ == '__main__':
    sys.exit(main())
