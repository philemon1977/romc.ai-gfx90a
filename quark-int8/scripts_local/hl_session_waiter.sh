#!/usr/bin/env bash
# 会话看守：每 5 分钟把状态摘要追加到一个文件；optimizer 进程消失（或写下终态 stop_reason）就退出。
# 设计成"后台作业"用：它只在会话结束时返回一次，避免父 agent 反复轮询烧上下文。
set -uo pipefail
RUN_TAG="${1:?用法: hl_session_waiter.sh <run_tag>}"
# trace 必须落在宿主可写的目录：session/optimizer_runs/ 是 root 属主（会话在容器里以 root 跑），
# 以 qiba 身份写它会 Permission denied 并让看守当场退出。
TRACE=/home/qiba/ROCm.AI/hyperloom/logs/watch_${RUN_TAG}.log
: > "$TRACE"
while true; do
  # 真实命令行是 "... cli --verbose optimize ..."，所以模式只能到模块名，
  # 不能写 "cli optimize" 相邻（那样永远匹配不到 => 看守会把活会话误判成已结束）。
  ALIVE=$(docker exec hyperloom-local bash -lc 'pgrep -c -f "[h]yperloom.inference_optimizer" || true')
  # 必须带 -i：docker exec 不接 stdin 时，heredoc 里的脚本根本进不到 python3，
  # 表现是"命令成功但输出为空"（第一次挂上时 trace 只有 alive=1 没有状态串）。
  ST=$(docker exec -i -e PYTHONPATH=/home/qiba/ROCm.AI/hyperloom hyperloom-local python3 - <<'PY' 2>/dev/null
import json,glob,os
c=sorted(glob.glob("/home/qiba/ROCm.AI/hyperloom/session/GLM-5.3-Flash-Quark-Int8/*/state.json"), key=os.path.getmtime)
if not c:
    print("no-state-yet"); raise SystemExit
d=json.load(open(c[-1]))
s=d.get("summary",d)
def g(k, default=None):
    return s.get(k, d.get(k, default))
cb = g("current_best", {}) or {}
best = cb.get("tput", cb.get("output_throughput")) if isinstance(cb, dict) else None
print("phase=%s baseline_tput=%s best=%s gain_validated=%s stop=%s stack=%d" % (
    g("phase"), g("baseline_tput"), best,
    g("cumulative_gain_validated"), g("stop_reason"),
    len(g("optimization_stack", []) or [])))
PY
)
  echo "$(date -u +%H:%M:%SZ) alive=${ALIVE:-?} $ST" >> "$TRACE"
  if [ "${ALIVE:-0}" = "0" ]; then
    echo "SESSION_GONE at $(date -u +%H:%M:%SZ)" >> "$TRACE"
    tail -25 "$TRACE"
    exit 0
  fi
  sleep 300
done
