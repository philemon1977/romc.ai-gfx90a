#!/bin/bash
# 改完即查（机制化"改完必检查"，替代靠记性）——起因：2026-09-20 我在三轮评审里
# 自己引入了 3 处语法错误（悬空 elif / 引号不配对 / 未定义变量），每次都是"跑一下"才发现。
# 用法：任何改动后 `bash check_my_code.sh`（不需要 GPU；守门测试在容器内跑，不碰卡）
set -u
P=${AI_HOME}/recipes/patches/vllm/vllm-openai-rocm-nightly-0918/core/tree
P_MODEL="${AI_HOME}/recipes/patches/vllm/vllm-openai-rocm-nightly-0918/DeepSeek/DeepSeek-V4.1-Flash-748B/tree"
Q=/home/qiba/ROCm.AI/quark-int8
bad=0
echo "== 1) 补丁文件严格编译（py_compile：能抓 ast.parse 抓不到的 global 顺序/未定义名） =="
for f in "${P_MODEL}/models/deepseek_v41/common/engram.py" "${P_MODEL}/models/deepseek_v41/amd/vl_model.py" \
         "$P/v1/attention/ops/rocm_aiter_mla_sparse.py" "$P/model_executor/layers/sparse_attn_indexer.py" \
         "$P/_aiter_ops.py" "$P/model_executor/layers/mhc.py"; do
  [ -f "$f" ] || continue
  if python3 -m py_compile "$f" 2>/dev/null; then echo "  ✓ $(basename "$f")"; else echo "  ✗ $(basename "$f")"; bad=1; fi
done
echo "== 2) 工具与测试脚本编译 =="
for f in fix_fp4_nibble_order fix_fp4_nibble_order_test audit_fp4_nibble_order verify_fixed_repo perf_bench \
         idx_layout_probe idx_kernel_v2_truth fact_recall_probe; do
  [ -f "$Q/$f.py" ] || continue
  if python3 -m py_compile "$Q/$f.py" 2>/dev/null; then echo "  ✓ $f.py"; else echo "  ✗ $f.py"; bad=1; fi
done
echo "== 3) shell 语法 =="
for f in "$Q"/scripts_local/*.sh; do
  if bash -n "$f" 2>/dev/null; then :; else echo "  ✗ $(basename "$f")"; bad=1; fi
done
[ "$bad" = 0 ] && echo "  ✓ 全部 shell 语法 OK"
echo "== 4) 守门测试（CPU-only，不占卡） =="
docker run --rm --entrypoint bash -v "$Q":/work vllm/vllm-openai-rocm:nightly-0918 \
  -c "python3 /work/fix_fp4_nibble_order_test.py" 2>&1 | tail -3 || bad=1
echo "== 结论：$([ "$bad" = 0 ] && echo 通过 || echo 有失败) =="
exit $bad
