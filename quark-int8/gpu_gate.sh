#!/bin/bash
# 起服前的"卡门"（可 source、可单测）。2026-09-21 从脚本里抽出来独立成文件，原因：
# 同一份判据在 qr_ab_watch.sh / stack_probe.sh 里各写一遍，今天写错了两次——
#   · ps -eo args | grep -F 'vllm.entrypoints.openai.api_server' 会匹配到 **grep 自己的命令行**
#     ⇒ 门永远不过（表现为"卡明明是空的却一直在等"）；
#   · [ "$el" -ge $${MAXWAIT:-7200} ] 里的 $$ 是 PID，后面 {MAXWAIT...} 成了字面量
#     ⇒ "integer expression expected"，超时判断整体失效。
# 判据（两条都要满足）：
#   1) 除"我们自己的端口"之外没有任何 vllm api_server 进程（不抢卡、不腾地方）；
#   2) 八张 GCD 每张 used < 5 GiB（与 launcher 的硬门同阈值）。
# 用法：
#   source quark-int8/gpu_gate.sh
#   gate 8121 && echo 可以起
#   wait_free 8121 7200        # 阻塞到放行；超时返回 1

gate(){
  local mine=${1:-8121} others
  # ★ pgrep + 方括号模式：模式文本不会被自己的命令行匹配上（pgrep 也天然排除自身）
  others=$(pgrep -af 'vllm.entrypoints.openai.api[_]server' 2>/dev/null \
             | grep -vF -- "--port $mine" | grep -c . || true)
  if [ "${others:-0}" != "0" ]; then
    echo "  gate: 检测到 ${others} 个别人的 api_server（端口非 $mine）" >&2
    return 1
  fi
  local h f u
  for h in 0 1 2 3 4 5 6 7; do
    f=/sys/class/drm/card$((h+1))/device/mem_info_vram_used
    u=$(cat "$f" 2>/dev/null || echo 0)
    if [ "${u:-0}" -gt $((5*1024*1024*1024)) ]; then
      echo "  gate: GCD$h 已用 $((u/1073741824)) GiB > 5 GiB" >&2
      return 1
    fi
  done
  return 0
}

vram_line(){ rocm-smi --showmeminfo vram 2>/dev/null | grep 'Used Memory' \
             | awk '{printf "%.2f ", $NF/1073741824}'; }

wait_free(){
  local mine=${1:-8121} max=${2:-7200} t0 last=0 el
  t0=$(date +%s)
  while :; do
    if gate "$mine" 2>/dev/null; then echo "  gate: 放行（$(vram_line)GiB）"; return 0; fi
    el=$(( $(date +%s) - t0 ))
    if [ "$el" -ge "$max" ]; then echo "  gate: 等待超时 ${max}s"; return 1; fi
    if [ $(( el - last )) -ge 300 ]; then
      last=$el
      echo "[gate] 仍在等（${el}s / ${max}s；$(vram_line)GiB）"
    fi
    sleep 30
  done
}
