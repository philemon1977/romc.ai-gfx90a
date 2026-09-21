#!/usr/bin/env bash
# 会话收场后的一次性读数：baseline / 优化栈 / 验证增益 / 终态原因 / 报告与产物路径。
# 设计原则：只读工具自己写的权威文件（state.json + reports/final.*），不拿中途快照凑数。
set -uo pipefail
S="${1:-/home/qiba/ROCm.AI/hyperloom/session/GLM-5.3-Flash-Quark-Int8/20260921T151223Z-34588c9d}"
echo "SESSION=$S"
echo "=== reports 目录 ==="
ls -l "$S/reports" 2>/dev/null | head -12 || echo "  (无 reports 目录)"
echo
echo "=== 权威状态（state.json）==="
docker exec -i hyperloom-local python3 - "$S" <<'PY'
import json, sys, time
s = json.load(open(sys.argv[1] + "/state.json"))
def g(k, d=""):
    return s.get(k, d)
print("stop_reason               =", repr(g("stop_reason")))
print("phase                     =", g("phase"))
print("baseline_tput             =", g("baseline_tput"))
print("cumulative_gain_validated =", g("cumulative_gain_validated"))
print("baseline_runtime_sec      =", g("baseline_runtime_sec"))
cb = g("current_best") or {}
if isinstance(cb, dict):
    print("current_best.action       =", cb.get("action"))
    print("current_best.args         =", (cb.get("effective_extra_server_args") or "")[:200])
    print("current_best.envs         =", str(cb.get("extra_envs"))[:220])
st = g("optimization_stack") or []
print("optimization_stack 条数   =", len(st))
for i, e in enumerate(st):
    if not isinstance(e, dict):
        print("  [%d] %r" % (i, e)); continue
    print("  [%d] action=%s accuracy=%s gain=%s" % (i, e.get("action"), e.get("accuracy"), e.get("gain_pct", e.get("delta_pct"))))
    print("      args = %s" % (e.get("effective_extra_server_args") or "")[:170])
    print("      envs = %s" % str(e.get("candidate_extra_envs"))[:220])
for h in (g("phase_history") or [])[-5:]:
    print("  hist:", str(h)[:190])
print("phase_elapsed_totals      =", g("phase_elapsed_totals"))
dl = float(g("deadline_unix") or 0)
print("deadline_unix             =", dl, ("(已过 %d 秒)" % int(time.time() - dl)) if dl and time.time() > dl else "")
PY
echo
echo "=== final.md 头部（若存在）==="
F="$S/reports/final.md"
if [ -f "$F" ]; then head -40 "$F"; else echo "  (final.md 还没写)"; fi
