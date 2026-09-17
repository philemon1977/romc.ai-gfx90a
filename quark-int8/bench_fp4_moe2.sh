#!/usr/bin/env bash
# The example's benchmark harness flushes the cache with 50 rotating buffers per iteration
# (stream_config arg 7 = flush_cache_), which dominated the "Perf" number (a constant
# ~18 ms even at 8 experts / 16 tokens => the reported 26 ms was NOT kernel time).
# Patch it out, rebuild, and re-measure the real kernel time at our model's shapes.
set -uo pipefail
OUT=/home/qiba/ROCm.AI/quark-int8/ck_int4
LOG=/home/qiba/ROCm.AI/quark-int8/logs/fp4_moe_bench2.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== fp4_moe_bench2 start $(date -u +%FT%TZ) ==="

docker run --rm --entrypoint bash -v "$OUT:/work" vllm/vllm-openai-rocm:nightly -c '
set -u
CK=/usr/local/lib/python3.12/dist-packages/aiter_meta/3rdparty/composable_kernel
SRC=$CK/example/ck_tile/18_flatmm
rm -rf /work/mp && mkdir -p /work/mp
cp $SRC/mixed_prec/a16w4_moe_flatmm.cpp $SRC/mixed_prec/a16w4_moe_flatmm.hpp \
   $SRC/mixed_prec/run_a16w4_moe_flatmm_example.inc $SRC/mixed_prec/a16w4_flatmm.hpp /work/mp/
python3 - <<PY
p = "/work/mp/run_a16w4_moe_flatmm_example.inc"
s = open(p).read()
old = "ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat, true, true, 50}"
new = "ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat, true, false, 50}"
assert old in s, "stream_config 行未匹配"
open(p, "w").write(s.replace(old, new))
print("  ✅ 已关闭 flush_cache_（第 7 位 true→false）")
PY
echo "=== 重编（gfx90a）==="
if hipcc -std=c++17 -O2 -DNDEBUG --offload-arch=gfx90a -DCK_ENABLE_BF16 -DCK_ENABLE_INT8 \
    -I$CK/include -I/work/mp -I$SRC -I$CK/library/include \
    /work/mp/a16w4_moe_flatmm.cpp -o /work/a16w4_moe_noflush.gfx90a > /work/noflush.build.log 2>&1; then
  echo "  ✅ 编译通过 $(stat -c%s /work/a16w4_moe_noflush.gfx90a) 字节"
else
  echo "  ❌ 编译失败："; grep "error:" /work/noflush.build.log | head -5
fi'
echo "--- 重测（关闭 flush，真实内核时间）---"
cd "$OUT"
for spec in "gemm1_gate_up 16 2048 4096" "gemm1_gate_up 64 2048 4096" "gemm2 16 4096 1024" "gemm2 64 4096 1024"; do
  set -- $spec
  printf "%-14s tokens=%-4s N=%-5s K=%-5s : " "$1" "$2" "$3" "$4"
  timeout 600 env LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib ./a16w4_moe_noflush.gfx90a \
    -gemm_kind=$1 -mixed_prec=bf16xfp4 -experts=512 -TopK=10 -NumTokens=$2 -N=$3 -K=$4 \
    -validate=0 -warmup=20 -repeat=50 2>&1 | grep -aoE "Perf: *[0-9.]+ ms[^,]*,[^,]*" | head -1
done
echo "=== fp4_moe_bench2 done $(date -u +%FT%TZ) ==="
