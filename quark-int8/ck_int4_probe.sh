#!/usr/bin/env bash
# Decisive numeric probe: patch the local CK int4 example to PRINT device results.
#   * init_method=1 (integer) -> A all 1.0, B nibbles alternate (1,0) => C(0,0) == K_logical/2
#     => a printed constant both (a) rules out a degenerate "all zeros == all zeros" pass and
#        (b) reveals whether the kernel's K counts packed units (K_logical = 2*K) or logical K.
#   * verify=1 also runs CK's host reference and check_err (which prints on failure only).
# NDEBUG is used because CK's host Tensor<pk_i4_t> indexing trips a too-strict assert.
set -uo pipefail
OUT=/home/qiba/ROCm.AI/quark-int8/ck_int4
LOG=/home/qiba/ROCm.AI/quark-int8/logs/ck_int4_probe.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== ck_int4_probe start $(date -u +%FT%TZ) ==="

docker run --rm --entrypoint bash -v "$OUT:/work" vllm/vllm-openai-rocm:nightly -c '
set -u
CK=/usr/local/lib/python3.12/dist-packages/aiter_meta/3rdparty/composable_kernel
for name in gemm_xdl_bf16_pk_i4_v3 gemm_xdl_fp16_pk_i4_v3_b_scale; do
  python3 - "$name" <<PY
import sys
name = sys.argv[1]
p = f"/work/{name}.cpp"
s = open(p).read()
assert "CKDBG" not in s, "already patched"
probe = """    c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());
    std::cout << "CKDBG dev(0,0)=" << static_cast<float>(c_m_n_device_result(0, 0))
              << " dev(0,1)=" << static_cast<float>(c_m_n_device_result(0, 1))
              << " dev(1,0)=" << static_cast<float>(c_m_n_device_result(1, 0))
              << " dev(7,13)=" << static_cast<float>(c_m_n_device_result(7, 13)) << std::endl;
    return pass;
"""
n = s.count("    return pass;")
assert n == 1, n
s = s.replace("    return pass;", probe)
open(p, "w").write(s)
print("  patched", p)
PY
  echo "=== $name ==="
  if hipcc -std=c++17 -O2 -DNDEBUG --offload-arch=gfx90a -DCK_ENABLE_BF16 \
      -I$CK/include -I$CK/library/include -I$CK/profiler/include -I$CK/example/01_gemm \
      /work/$name.cpp $CK/library/src/utility/device_memory.cpp $CK/library/src/utility/host_tensor.cpp \
      -o /work/$name.probe > /work/$name.probe.log 2>&1; then
    echo "  ✅ 编译通过"
  else
    echo "  ❌ 编译失败："; grep "error:" /work/$name.probe.log | head -4; continue
  fi
done'
echo "--- 运行 ---"
cd "$OUT"
for b in gemm_xdl_bf16_pk_i4_v3.probe gemm_xdl_fp16_pk_i4_v3_b_scale.probe; do
  [ -x "$OUT/$b" ] || { echo "缺 $b"; continue; }
  echo "################ $b"
  for spec in "1 1 0 256 2048 4096" "1 2 0 256 2048 4096" "1 2 0 3840 4096 4096"; do
    echo "---- verify=1 init/time/M/N/K = $spec  （init=1 时 C 应为常数 K_logical/2）"
    timeout 300 env LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib ./$b $spec -1 -1 -1 1 2>&1 | grep -aE "CKDBG|Error|Perf|assert" | head -4
  done
done
echo "=== ck_int4_probe done $(date -u +%FT%TZ) ==="
