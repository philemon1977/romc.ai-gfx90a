#!/usr/bin/env bash
# hyperloom-fa 容器内：AITER ASM 注意力 gfx90a 移植（aiter-cdna2 工具链 → envs/vllm-fa）
# 步骤按 launcher/beta/gfx90a-aiter-cdna2/docs/port-matrix.md §Reproducing
set -uo pipefail
P=/opt/envs/vllm/lib/python3.12/site-packages
export PATH=/opt/envs/vllm/bin:$PATH
cd /opt/aiter-cdna2

echo "== [1/6] classify+repatch gfx942->gfx90a (242-kernel ceiling)"
python3 - <<EOF
import sys; sys.path.insert(0,"/opt/envs/vllm/lib/python3.12/site-packages")
import aiter, subprocess
print("aiter ok:", aiter.__file__)
EOF
python3 tools/repatch_gfx942_to_gfx90a.py "$P/aiter_meta/hsa/gfx942" /tmp/port_v2 2>&1 | tail -3

echo "== [2/6] install generated hsa/gfx90a (backup original first)"
if [ -d "$P/aiter_meta/hsa/gfx90a" ]; then echo "gfx90a dir exists, keeping"; else
  [ -e "$P/aiter_meta/hsa/gfx90a" ] || cp -a /tmp/port_v2 "$P/aiter_meta/hsa/gfx90a" && echo installed; fi
ls "$P/aiter_meta/hsa/gfx90a" | head -4

echo "== [3/6] open aiter ASM arch gates (SITE retarget py3.12)"
mkdir -p /tmp/fa_patch
sed 's|/opt/python/lib/python3.14/site-packages|/opt/envs/vllm/lib/python3.12/site-packages|' \
  patches/enable_gfx90a_asm_paths.py > /tmp/fa_patch/enable_gfx90a_asm_paths.py
python3 /tmp/fa_patch/enable_gfx90a_asm_paths.py 2>&1 | tail -5

echo "== [4/6] open vLLM attention gates (SITE retarget, vllm 0.28)"
sed 's|/opt/python/lib/python3.14/site-packages|/opt/envs/vllm/lib/python3.12/site-packages|' \
  patches/enable_vllm_aiter_gfx90a.py > /tmp/fa_patch/enable_vllm_aiter_gfx90a.py
python3 /tmp/fa_patch/enable_vllm_aiter_gfx90a.py 2>&1 | tail -5

echo "== [5/6] purge stale JIT modules"
rm -f "$P/aiter/jit/module_fmha_v3_fwd.so" "$P/aiter/jit/module_pa_fwd"*.so 2>/dev/null
rm -rf "$P/aiter/jit/build/module_fmha_v3_fwd" "$P/aiter/jit/build/module_pa"*.so 2>/dev/null
echo purged

echo "== [6/6] validate: FMHA ASM (build ~2min) + PA"
AITER_LOG_LEVEL=info python3 tests/test_fmha_v3_fwd_asm_gfx90a.py --require-asm 2>&1 | tail -4
AITER_LOG_LEVEL=info python3 tests/test_pa_fwd_asm_gfx90a.py 2>&1 | tail -4
echo "FA_PORT_DONE rc=$?"
