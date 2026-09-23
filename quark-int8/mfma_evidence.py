# -*- coding: utf-8 -*-
"""证明 vLLM 的 int4(W4A16) 内核在 gfx90a 上走 MFMA 矩阵核。

vLLM 侧事实（镜像内 grep）：
  .../kernels/linear/mixed_precision/triton_w4a16.py:154  accumulator += tl.dot(a, b_fp, out_dtype=tl.float32)
  .../layers/fused_moe/fused_moe.py:275/538/...           accumulator = tl.dot(a, b, acc=accumulator)
即 int4 权重在内核内 dequant 成 bf16，再做 tl.dot。
本脚本用 gfx90a target 编译同样形状/dtype 的 tl.dot，检查生成汇编里是否出现 v_mfma。
"""
import re
import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton.compiler.compiler import GPUTarget


@triton.jit
def _dot_kernel(a_ptr, b_ptr, c_ptr):
    rm = tl.arange(0, 16); rn = tl.arange(0, 16); rk = tl.arange(0, 16)
    a = tl.load(a_ptr + rm[:, None] * 16 + rk[None, :])   # bf16 激活
    b = tl.load(b_ptr + rk[:, None] * 16 + rn[None, :])   # dequant 出来的 bf16 权重
    acc = tl.dot(a, b, out_dtype=tl.float32)
    tl.store(c_ptr + rm[:, None] * 16 + rn[None, :], acc)


def scan():
    src = ASTSource(fn=_dot_kernel,
                    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*fp32"})
    tgt = GPUTarget("hip", "gfx90a", 64)
    cc = triton.compile(src, target=tgt)
    asm = cc.asm.get("amdgcn", "") or cc.asm.get("ptx", "")
    mfma = sorted(set(re.findall(r"v_mfma[a-z0-9_]*", asm)))
    fma = sorted(set(re.findall(r"v_fma[a-z0-9_]*|v_pk_fma[a-z0-9_]*", asm)))
    print("  target=gfx90a  kernel=tl.dot(16x16x16, bf16 x bf16 -> fp32)")
    print("    MFMA 指令: %s" % (", ".join(mfma) if mfma else "（无）"))
    print("    FMA 指令 : %s" % (", ".join(fma) if fma else "（无）"))
    n = asm.count("v_mfma")
    print("    v_mfma 出现次数: %d" % n)
    return mfma


print("=== bf16 tl.dot 在 gfx90a 的降级结果 ===")
m = scan()
print("=== 结论 ===")
print("  int4 权重路径（tl.dot）落到 MFMA 矩阵核: %s" % ("是" if m else "否"))
