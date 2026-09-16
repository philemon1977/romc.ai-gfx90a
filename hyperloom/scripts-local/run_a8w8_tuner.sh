#!/usr/bin/env bash
# A：为 gfx90a 调优 AITER a8w8，取每形状最优 kernel。
#
# 依据：
#  - 调优器用法见 aiter_meta/csrc/ck_gemm_a8w8/README.md；`-k` 启用 splitK 候选。
#  - 查找表由 tuned CSV 在**编译期**生成（GENERATE_LOOKUP_TABLE），所以调优结果会
#    直接进入 rowwise_dispatch 的第一步，不再依赖启发式 —— 这正是本次 +17.6% 的杠杆所在。
#  - 形状清单只取 2 的幂档位：未命中时派发逻辑会把 M 向上取到 2 的幂再查一次
#    （padded_m = nextPow2(M)；1<M<=16 -> 16），故 {1,16,32,64,128,256,512,2048} 覆盖整段批量。
#
# 前置：必须在**没有其它 GPU 负载**时跑（它会占一张卡做 benchmark）。
# 用法（容器内）：bash run_a8w8_tuner.sh
set -uo pipefail

SP=/opt/envs/wu1w/lib/python3.12/site-packages
OUT=/home/qiba/ROCm.AI/hyperloom/reports/models/qwen38-27b-w8a8-dense/tuning
IN=/home/qiba/ROCm.AI/hyperloom/scripts-local/a8w8_untuned_gfx90a.csv
mkdir -p "$OUT"

# 脚本内部用相对路径（aiter/configs/...），必须从 site-packages 目录启动
cd "$SP"

unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export PYTHONPATH=/home/qiba/ROCm.AI/hyperloom/envs/usersite-shim/lib/python3.12/site-packages

echo "== 调优开始 $(date -u +%FT%TZ)  输入 $IN（$(( $(wc -l < "$IN") - 1 )) 个形状）=="
/opt/envs/wu1w/bin/python -u aiter_meta/csrc/ck_gemm_a8w8/gemm_a8w8_tune.py \
  -i "$IN" \
  -o "$OUT/a8w8_tuned_gfx90a.csv" \
  -k \
  --profile_file "$OUT/profile_a8w8_all_candidates.csv" \
  -v
RC=$?
echo "== 调优结束 rc=$RC $(date -u +%FT%TZ) =="
echo "--- 胜者行数: $(( $(wc -l < "$OUT/a8w8_tuned_gfx90a.csv" 2>/dev/null || echo 1) - 1 )) ---"
exit $RC
