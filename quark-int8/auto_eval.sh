#!/bin/bash
# 无人值守端到端：等模型就绪 → 起【新】模型评测 → 停 → 起【旧】模型取基线 → 停 → 出对比。
#
# 为什么串成一条：目标轮次有限，必须让整条链自己跑完，我只需读结果。
# 优先级：先跑【新】（要回答的问题），再跑【旧】（同尺子的基线，且只跑固定题组省时间）。
set -u
OUT=${OUT:-/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4-opt-allbf16}
OLD=/home/qiba/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4
REPO=/home/qiba/ROCm.AI/quark-int8
CL=${CL:?需传入收尾日志路径}
echo "chain log = $CL"

echo "=== $(date +%T) 等 engram requant 完成 ==="
for i in $(seq 1 480); do
  if grep -qa "完毕" "$CL" 2>/dev/null; then echo "✅ $(date +%T) 收尾完成"; break; fi
  sleep 30
done
tail -14 "$CL"
n=$(ls "$OUT"/model-000*-of-00048.safetensors 2>/dev/null | wc -l)
sz=$(du -sh "$OUT" 2>/dev/null | cut -f1)
echo "分片数=$n 体积=$sz（预期 48 / ≈399 GiB）"
if [ "$n" -ne 48 ]; then echo "❌ 分片不全，终止"; exit 1; fi

cd "$REPO"

# ★ 两臂必须**同题集、同参数**：`degenerate_rate = 退化题数 / 总题数`，
#   若一臂跑 9 题、另一臂跑 5 题，两个分母不同的比率就被直接对比 ✗
#   （正是 CLAUDE.md §9「一项修复、两把尺子 ≠ 两笔成果」要防的事）。
#   原计划给旧臂设 NGSM8K=0 以省 5 分钟 ⇒ 会污染主结论 ⇒ 改为两臂同为 4 ✓
NGSM8K_BOTH=${NGSM8K:-4}
echo "=== $(date +%T) 起【新】模型评测（GSM8K=$NGSM8K_BOTH）==="
LABEL=new MODEL_PATH="$OUT" NGSM8K="$NGSM8K_BOTH" bash serve_and_eval.sh || echo "⚠️ 新臂失败（继续跑旧臂，但对比会缺一侧）"

echo "=== $(date +%T) 起【旧】模型取基线（同题集同参数，GSM8K=$NGSM8K_BOTH）==="
LABEL=old MODEL_PATH="$OLD" NGSM8K="$NGSM8K_BOTH" bash serve_and_eval.sh || echo "⚠️ 旧臂失败"

echo "=== $(date +%T) 全部完毕；A/B 对比 ==="
NEWJ=$(ls -t "$REPO"/logs/quality_new_*.json 2>/dev/null | head -1)
OLDJ=$(ls -t "$REPO"/logs/quality_old_*.json 2>/dev/null | head -1)
echo "new=$NEWJ"; echo "old=$OLDJ"
echo "注：degenerate_rate / gsm8k_hit_rate / needle_hit 两臂同尺可比 ✓；"
echo "    tok_per_s **不可跨臂比较**（两臂起服时刻与机器负载不同），TPS 交付值以旧 checkpoint 256K 那次为准。"
[ -n "$NEWJ" ] && [ -n "$OLDJ" ] && python3 "$REPO/eval_quality_ab.py" --compare "$OLDJ" "$NEWJ"
echo "=== $(date +%T) 链结束 ==="
