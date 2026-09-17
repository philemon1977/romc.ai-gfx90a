#!/usr/bin/env bash
# S2 step 2: the ck_tile flatmm W4A16 MoE example only WIRES pk_fp4_t (fp16xfp4 / bf16xfp4);
# its kernel template accepts pk_int4_t (see the static_asserts) but there is no int4 branch
# (the else throws "Unsupported precision type"). Add bf16xi4 branches and try to build for
# gfx90a — a compile is already informative, a run would be decisive.
set -uo pipefail
OUT=/home/qiba/ROCm.AI/quark-int8/ck_int4
mkdir -p "$OUT"
docker run --rm --entrypoint bash -v "$OUT:/work" vllm/vllm-openai-rocm:nightly -c '
set -u
CK=/usr/local/lib/python3.12/dist-packages/aiter_meta/3rdparty/composable_kernel
D=$CK/example/ck_tile/18_flatmm/mixed_prec
cp $D/a16w4_moe_flatmm.cpp /work/a16w4_moe_i4.cpp
python3 - <<PY
import re
p = "/work/a16w4_moe_i4.cpp"
s = open(p).read()
# gemm1_gate_up: insert int4 branch right before the throwing else
g1_fp4 = """            else if(mixed_prec == "bf16xfp4")
            {
                return run_a16w4_moe_gemm_example_with_layouts<
                    ck_tile::bfloat16_t,
                    ck_tile::pk_fp4_t,
                    FlatmmConfig,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up>(argc, argv, Row{}, Col{}, Row{});
            }
"""
g1_i4 = g1_fp4.replace("pk_fp4_t", "pk_int4_t").replace("\"bf16xfp4\"", "\"bf16xi4\"")
assert g1_fp4 in s, "gemm1 fp4 branch not found"
s = s.replace(g1_fp4, g1_fp4 + g1_i4, 1)

g2_fp4 = """            else if(mixed_prec == "bf16xfp4")
            {
                return run_a16w4_moe_gemm_example_with_layouts<ck_tile::bfloat16_t,
                                                               ck_tile::pk_fp4_t,
                                                               FlatmmConfig,
                                                               ck_tile::MoeFlatmmKind::kFFN_gemm2>(
                    argc, argv, Row{}, Col{}, Row{});
            }
"""
g2_i4 = g2_fp4.replace("pk_fp4_t", "pk_int4_t").replace("\"bf16xfp4\"", "\"bf16xi4\"")
if g2_fp4 in s:
    s = s.replace(g2_fp4, g2_fp4 + g2_i4, 1)
    print("  gemm2 int4 分支已加")
else:
    print("  ⚠️ gemm2 的 fp4 分支文本未精确匹配，只加了 gemm1")
open(p, "w").write(s)
print("  int4 分支插入完成")
PY
echo "=== 编译（gfx90a）==="
if hipcc -std=c++17 -O2 -DNDEBUG --offload-arch=gfx90a -DCK_ENABLE_BF16 -DCK_ENABLE_INT8 \
    -I$CK/include -I$D -I$CK/example/ck_tile/18_flatmm -I$CK/library/include \
    /work/a16w4_moe_i4.cpp -o /work/a16w4_moe_i4.gfx90a > /work/a16w4_moe_i4.build.log 2>&1; then
  echo "✅ 编译通过 $(stat -c%s /work/a16w4_moe_i4.gfx90a) 字节"
  roc-obj-ls /work/a16w4_moe_i4.gfx90a 2>/dev/null | grep -oE "gfx[0-9a-f]+" | sort -u | tr "\n" " "; echo
else
  echo "❌ 编译失败；error 前 10 条："
  grep -E "error:" /work/a16w4_moe_i4.build.log | sed "s/^/   /" | sort -u | head -10
fi
echo "=== 用法（看它要什么参数）==="
grep -nE "mixed_prec|gemm_kind" $D/a16w4_moe_flatmm.cpp | grep -iE "get_str|default|argv" | head -6
' 2>&1 | tail -25
echo "=== build_ck_int4_moe_i4 done $(date -u +%FT%TZ) ==="
