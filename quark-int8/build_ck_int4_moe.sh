#!/usr/bin/env bash
# S2 step 1 (the S1 playbook): CK ships a ck_tile W4A16 MoE example
#   example/ck_tile/18_flatmm/mixed_prec/a16w4_moe_flatmm.cpp   (A=16-bit, W=int4, XDL, has main)
# Try to build it for gfx90a. Note the differing design from the 2-stage aiter MoE: the
# flatmm MoE is a different kernel family (MoeFlatmmKernel), so it may sidestep the
# bf16-global-atomic requirement that blocks CK's classic device_moe_gemm on CDNA2
# (device_prop.hpp: is_bf16_atomic_supported() == false on gfx90a, no software fallback).
set -uo pipefail
OUT=/home/qiba/ROCm.AI/quark-int8/ck_int4
mkdir -p "$OUT"
docker run --rm --entrypoint bash -v "$OUT:/work" vllm/vllm-openai-rocm:nightly -c '
set -u
CK=/usr/local/lib/python3.12/dist-packages/aiter_meta/3rdparty/composable_kernel
D=$CK/example/ck_tile/18_flatmm/mixed_prec
F=$D/a16w4_moe_flatmm.cpp
echo "=== 目标: $F ==="
for INC in "-I$CK/include -I$D -I$CK/example/ck_tile/18_flatmm -I$CK/library/include" \
           "-I$CK/include -I$D -I$CK/example/ck_tile/18_flatmm -I$CK/library/include -DCK_TILE_FMHA_FWD_FAST_EXP2=1"; do
  echo "--- 尝试包含路径: ${INC:0:80}..."
  if hipcc -std=c++17 -O2 -DNDEBUG --offload-arch=gfx90a -DCK_ENABLE_BF16 -DCK_ENABLE_INT8 \
      $INC "$F" -o /work/a16w4_moe_flatmm.gfx90a > /work/a16w4_moe_flatmm.build.log 2>&1; then
    echo "  ✅ 编译通过 $(stat -c%s /work/a16w4_moe_flatmm.gfx90a) 字节"
    roc-obj-ls /work/a16w4_moe_flatmm.gfx90a 2>/dev/null | grep -oE "gfx[0-9a-f]+" | sort -u | tr "\n" " "; echo
    exit 0
  fi
  echo "  ❌ 失败，前几条 error："
  grep -E "error:" /work/a16w4_moe_flatmm.build.log | sed "s/^/     /" | sort -u | head -8
done
echo "=== 全部尝试失败 ==="
' 2>&1 | tail -30
echo "=== build_ck_int4_moe done $(date -u +%FT%TZ) ==="
