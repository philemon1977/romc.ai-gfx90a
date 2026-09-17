#!/usr/bin/env bash
# DECISIVE EXPERIMENT: does CK's int4 (pk_i4) W4A16 GEMM compile and run on gfx90a?
#
# Background correction: CK DOES ship int4 GEMM device ops + examples
#   example/01_gemm/gemm_xdl_bf16_pk_i4_v3.cpp          (A=bf16, B=pk_i4  = W4A16)
#   example/01_gemm/gemm_xdl_fp16_pk_i4_v3_b_scale.cpp  (A=fp16, B=pk_i4 + B scale = W4A16 gs)
# and aiter's own MoE codegen already has `using I4 = ck::pk_i4_t;` + `MulABScaleWint4`.
# But the examples carry a HOST-side guard refusing anything except gfx942/950/gfx11/gfx12.
# There is no __gfx* guard inside device_gemm_xdl_cshuffle_v3*.hpp, so the guard may be
# conservative. This job (a) removes it and (b) compiles for gfx90a.
# Running the binary is deliberately deferred: the GPU is busy with the MTP sweep, and a
# bad int4 device path could hang the GCD and wreck in-flight measurements.
set -uo pipefail
OUT=/home/qiba/ROCm.AI/quark-int8/ck_int4
mkdir -p "$OUT"
docker run --rm --entrypoint bash -v "$OUT:/work" vllm/vllm-openai-rocm:nightly -c '
set -u
M=/usr/local/lib/python3.12/dist-packages/aiter_meta
CK=$M/3rdparty/composable_kernel
patch_gate () {  # src dst
  python3 - "$1" "$2" <<PY
import re, sys
s = open(sys.argv[1]).read()
s2, n = re.subn(r"if\(!\(ck::get_device_name\(\) == \"gfx942\".*?is_gfx12_supported\(\)\)\)",
                "if(false)", s, flags=re.S)
open(sys.argv[2], "w").write(s2)
print(f"  arch-gate removed: {n} occurrence(s)")
PY
}
for name in gemm_xdl_bf16_pk_i4_v3 gemm_xdl_fp16_pk_i4_v3_b_scale; do
  SRC=$CK/example/01_gemm/$name.cpp
  [ -f "$SRC" ] || { echo "缺 $SRC"; continue; }
  echo "=== $name ==="
  patch_gate "$SRC" "/work/$name.cpp"
  echo -n "  编译中 (hipcc --offload-arch=gfx90a) ... "
  if hipcc -std=c++17 -O2 --offload-arch=gfx90a -DCK_ENABLE_BF16 \
      -I$CK/include -I$CK/library/include -I$CK/profiler/include -I$CK/example/01_gemm \
      "/work/$name.cpp" \
      $CK/library/src/utility/device_memory.cpp \
      $CK/library/src/utility/host_tensor.cpp \
      -o "/work/$name.gfx90a" > "/work/$name.build.log" 2>&1; then
    echo "✅ 编译通过  $(stat -c%s /work/$name.gfx90a) 字节"
    echo -n "     码对象 arch: "
    (roc-obj-ls "/work/$name.gfx90a" 2>/dev/null | grep -oE "gfx[0-9a-f]+" | sort -u | tr "\n" " ") || echo "?"
    echo
  else
    echo "❌ 编译失败 —— 关键错误："
    grep -E "error:" "/work/$name.build.log" | sed "s/^/     /" | sort -u | head -6
  fi
done
echo
echo "=== 产物 ==="
ls -la /work/*.gfx90a 2>/dev/null || echo "  （无）"
' 2>&1
echo "=== build done $(date -u +%FT%TZ) ==="
