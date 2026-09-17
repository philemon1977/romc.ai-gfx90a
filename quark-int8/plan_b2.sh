#!/usr/bin/env bash
# PLAN B, step 2: the dispatcher call itself.
#
# step 1 (passing ComputeDataType into FlatmmPipelineProblem) was necessary but not sufficient:
#   include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp:508
#     using WarpGemm = WarpGemmDispatcher<typename Problem::ADataType,
#                                         typename Problem::BDataType,     <-- pk_int4_t => undefined
#                                         typename Problem::CDataType, ...>;
#   Problem::ComputeDataType exists (gemm_pipeline_problem.hpp:406). For the plain bf16 case
#   ComputeDataType == ADataType == BDataType, so substituting it here is behaviour-preserving;
#   for pk_int4 it makes the warp gemm resolve to the bf16 one (int4 is unpacked to the compute
#   type upstream), which is exactly how sample 03_gemm handles int4.
# Because every docker run is fresh, the header patch and the build happen in one invocation.
set -uo pipefail
OUT=/home/qiba/ROCm.AI/quark-int8/ck_int4
LOG=/home/qiba/ROCm.AI/quark-int8/logs/plan_b2.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== plan_b2 start $(date -u +%FT%TZ) ==="

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
assert old_prob in s
s = s.replace(old_prob, new_prob, 1)
g1 = """            else if(mixed_prec == "bf16xfp4")
            {
                return run_a16w4_moe_gemm_example_with_layouts<
                    ck_tile::bfloat16_t,
                    ck_tile::pk_fp4_t,
                    FlatmmConfig,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up>(argc, argv, Row{}, Col{}, Row{});
            }
"""
assert g1 in s
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
assert g2 in s
s = s.replace(g2, g2 + g2.replace("pk_fp4_t", "pk_int4_t").replace("bf16xfp4", "bf16xi4"), 1)
open(p, "w").write(s)
print("  ✅ 例程补丁完成（compute dtype + bf16xi4 分派）")
PY
# --- 关键修补：策略里的 dispatcher 实参改用 ComputeDataType ---
POL=$CK/include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp
cp $POL /work/policy.orig.hpp
python3 - <<PY
p = "/usr/local/lib/python3.12/dist-packages/aiter_meta/3rdparty/composable_kernel/include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp"
s = open(p).read()
old = """        using WarpGemm   = WarpGemmDispatcher<typename Problem::ADataType,
                                              typename Problem::BDataType,"""
new = """        using WarpGemm   = WarpGemmDispatcher<typename Problem::ADataType,
                                              typename Problem::ComputeDataType,"""
assert old in s, "policy dispatcher 调用点未匹配"
open(p, "w").write(s.replace(old, new, 1))
print("  ✅ (关键) policy: dispatcher 的 B 实参 BDataType -> ComputeDataType")
PY
echo "=== 编译（gfx90a）==="
if hipcc -std=c++17 -O2 -DNDEBUG --offload-arch=gfx90a -DCK_ENABLE_BF16 -DCK_ENABLE_INT8 \
    -I$CK/include -I/work/mpB -I$SRC -I$CK/library/include \
    /work/mpB/a16w4_moe_flatmm.cpp -o /work/a16w4_moe_i4B.gfx90a > /work/mpB.build.log 2>&1; then
  echo "  ✅ 编译通过 $(stat -c%s /work/a16w4_moe_i4B.gfx90a) 字节"
  roc-obj-ls /work/a16w4_moe_i4B.gfx90a 2>/dev/null | grep -oE "gfx[0-9a-f]+" | sort -u | tr "\n" " "; echo
else
  echo "  ❌ 编译失败；error 前 12 条（去重）："
  grep -E "error:" /work/mpB.build.log | sed "s/^/   /" | sort -u | head -12
fi'
echo "=== plan_b2 done $(date -u +%FT%TZ) ==="
