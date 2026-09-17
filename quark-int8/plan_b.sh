#!/usr/bin/env bash
# PLAN B: give ck_tile's flatmm MoE an int4 (bf16 x pk_int4) path.
#
# Root cause located (no new warp gemm needed):
#   example/ck_tile/18_flatmm/mixed_prec/a16w4_moe_flatmm.cpp
#     :97   using ComputeDataType = ADataType;                       // bf16  <- already defined
#     :121  conditional_t<MXFP4_Pipeline,
#              F16xMXF4FlatmmPipelineProblem<A, B, Acc, ...>,          // fp4 branch: passes compute dtype through
#              FlatmmPipelineProblem<A, B, Acc, ...>>;                 // int4 branch: B stays pk_int4_t
#   and gemm_pipeline_problem.hpp:389 declares FlatmmPipelineProblem's LAST parameter as
#     typename ComputeDataType_ = ADataType_
#   => the int4 branch simply never passes it, so the warp-gemm dispatcher is asked for
#      (bf16, pk_int4) which has no specialization (and none is needed: int4 is meant to be
#      unpacked to the compute type, exactly as sample 03_gemm does via AComputeDataType).
#
# This patch (a) adds a bf16xi4 dispatch and (b) passes ComputeDataType as the last template arg.
set -uo pipefail
OUT=/home/qiba/ROCm.AI/quark-int8/ck_int4
LOG=/home/qiba/ROCm.AI/quark-int8/logs/plan_b.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== plan_b start $(date -u +%FT%TZ) ==="

docker run --rm --entrypoint bash -v "$OUT:/work" vllm/vllm-openai-rocm:nightly -c '
set -u
CK=/usr/local/lib/python3.12/dist-packages/aiter_meta/3rdparty/composable_kernel
SRC=$CK/example/ck_tile/18_flatmm
rm -rf /work/mpB && mkdir -p /work/mpB
cp $SRC/mixed_prec/a16w4_moe_flatmm.cpp $SRC/mixed_prec/a16w4_moe_flatmm.hpp \
   $SRC/mixed_prec/run_a16w4_moe_flatmm_example.inc $SRC/mixed_prec/a16w4_flatmm.hpp /work/mpB/
python3 - <<PY
p = "/work/mpB/a16w4_moe_flatmm.cpp"
s = open(p).read()

# (1) pass ComputeDataType into the non-MXFP4 FlatmmPipelineProblem
old_prob = """                               ck_tile::FlatmmPipelineProblem<ADataType,
                                                              BDataType,
                                                              AccDataType,
                                                              CodegenFlatmmShape,
                                                              CodegenGemmTraits,
                                                              scheduler,
                                                              has_hot_loop_v,
                                                              tail_number_v>>;"""
new_prob = """                               ck_tile::FlatmmPipelineProblem<ADataType,
                                                              BDataType,
                                                              AccDataType,
                                                              CodegenFlatmmShape,
                                                              CodegenGemmTraits,
                                                              scheduler,
                                                              has_hot_loop_v,
                                                              tail_number_v,
                                                              ck_tile::amd_buffer_coherence_enum::coherence_default,
                                                              false,
                                                              ComputeDataType>>;"""
assert old_prob in s, "FlatmmPipelineProblem 调用点未匹配"
s = s.replace(old_prob, new_prob, 1)
print("  ✅ (1) FlatmmPipelineProblem 末位补上 ComputeDataType")

# (2) add bf16xi4 dispatch branches (gemm1_gate_up + gemm2)
g1 = """            else if(mixed_prec == "bf16xfp4")
            {
                return run_a16w4_moe_gemm_example_with_layouts<
                    ck_tile::bfloat16_t,
                    ck_tile::pk_fp4_t,
                    FlatmmConfig,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up>(argc, argv, Row{}, Col{}, Row{});
            }
"""
assert g1 in s, "gemm1 fp4 分支未匹配"
s = s.replace(g1, g1 + g1.replace("pk_fp4_t", "pk_int4_t").replace("bf16xfp4", "bf16xi4"), 1)

g2 = """            else if(mixed_prec == "bf16xfp4")
            {
                return run_a16w4_moe_gemm_example_with_layouts<ck_tile::bfloat16_t,
                                                               ck_tile::pk_fp4_t,
                                                               FlatmmConfig,
                                                               ck_tile::MoeFlatmmKind::kFFN_gemm2>(
                    argc, argv, Row{}, Col{}, Row{});
            }
"""
assert g2 in s, "gemm2 fp4 分支未匹配"
s = s.replace(g2, g2 + g2.replace("pk_fp4_t", "pk_int4_t").replace("bf16xfp4", "bf16xi4"), 1)
print("  ✅ (2) bf16xi4 分派已加入 gemm1/gemm2")

open(p, "w").write(s)
PY
echo "=== 编译（gfx90a）==="
if hipcc -std=c++17 -O2 -DNDEBUG --offload-arch=gfx90a -DCK_ENABLE_BF16 -DCK_ENABLE_INT8 \
    -I$CK/include -I/work/mpB -I$SRC -I$CK/library/include \
    /work/mpB/a16w4_moe_flatmm.cpp -o /work/a16w4_moe_i4B.gfx90a > /work/mpB.build.log 2>&1; then
  echo "  ✅ 编译通过 $(stat -c%s /work/a16w4_moe_i4B.gfx90a) 字节"
  roc-obj-ls /work/a16w4_moe_i4B.gfx90a 2>/dev/null | grep -oE "gfx[0-9a-f]+" | sort -u | tr "\n" " "; echo
else
  echo "  ❌ 编译失败；error 前 12 条："
  grep -E "error:" /work/mpB.build.log | sed "s/^/   /" | sort -u | head -12
fi'
echo "=== plan_b done $(date -u +%FT%TZ) ==="
