#!/usr/bin/env bash
# 多会话共卡守卫（source 本文件即可）。规矩来自用户 2026-09-18：
#   ① 不抢 GPU  ② 不杀别人起的服务  ③ 等对方释放后才自己起服务
# 机制（而不是靠自觉）：
#   * 本会话用**专用端口**（默认 8127）⇒ PID 文件 / 日志名都与并行会话的 8117 隔离，
#     既不会覆盖对方的 PID 文件，也不会与对方同分钟撞日志名（事故：两会话同用 8117，
#     同一份 ornith397b-8117.pid 被互相覆盖，导致我按 PID 文件停服时杀掉了对方的服务）。
#   * 停服**只杀本文件记录过的 pid**（$PID_FILE 是本会话独占路径），并且杀前核对
#     /proc/<pid>/cmdline 里确实是 api_server 且端口是本会话端口。
#   * 起服前**硬门**：任何别的 api_server 进程存在，或任何 die 显存不足 ⇒ 直接退出并报告，
#     绝不尝试"腾地方"。
set -uo pipefail

MY_PORT="${MY_PORT:-8127}"
LOG_ROOT="${LOG_ROOT:-/home/qiba/ai/logs}"
export PORT="$MY_PORT"
export PID_FILE="${PID_FILE:-${LOG_ROOT}/ornith397b-${MY_PORT}-dsh.pid}"
export LOG_FILE="${LOG_FILE:-${LOG_ROOT}/ornith397b/server-${MY_PORT}-dsh-$(date +%Y%m%d-%H%M).log}"
VRAM_MIN_GIB="${VRAM_MIN_GIB:-62}"

_guard_log () { echo "[guard] $*"; }

guard_no_other_servers () {
  local others
  others=$(pgrep -af "vllm.entrypoints.openai.api_server" 2>/dev/null | grep -v -- "--port ${MY_PORT}\b" || true)
  if [ -n "$others" ]; then
    _guard_log "⛔ 检测到其它会话的 vLLM 服务（不抢不杀，退出）："
    echo "$others" | sed 's/^/    /' | cut -c1-160
    _guard_log "   对方释放后再跑；或与对方约定各自的端口/PID 文件。"
    return 1
  fi
  return 0
}

guard_vram_free () {
  local h free worst=999
  for h in 0 1 2 3 4 5 6 7; do
    local c="/sys/class/drm/card$((h+1))/device/mem_info_vram_used"
    [ -r "$c" ] || { _guard_log "⛔ 读不到 $c（fail-closed）"; return 1; }
    free=$(( (68702699520 - $(cat "$c")) / 2**30 ))
    [ "$free" -lt "$worst" ] && worst=$free
  done
  if [ "$worst" -lt "$VRAM_MIN_GIB" ]; then
    _guard_log "⛔ 最小的 die 只剩 ${worst} GiB（要求 ${VRAM_MIN_GIB}）⇒ 有主或未释放，退出"
    return 1
  fi
  _guard_log "8 die 全空（最小 ${worst} GiB）✅"
  return 0
}

guard_port_free () {
  if nc -z 127.0.0.1 "$MY_PORT" 2>/dev/null; then
    _guard_log "⛔ 本会话端口 $MY_PORT 已被占用（可能是残留或别人也在用这个号）⇒ 退出"
    return 1
  fi
  return 0
}

guard_all () { guard_no_other_servers && guard_vram_free && guard_port_free; }

# 只停本会话自己的服务：先核对 cmdline 里的端口，再按进程组 TERM
stop_mine () {
  [ -f "$PID_FILE" ] || { _guard_log "无本会话 PID 文件（$PID_FILE），不动任何进程"; return 0; }
  local p; p=$(cat "$PID_FILE" 2>/dev/null || true)
  [ -n "${p:-}" ] || { rm -f "$PID_FILE"; return 0; }
  if ! kill -0 "$p" 2>/dev/null; then
    _guard_log "PID 文件里的 $p 已不在（陈旧）⇒ 删文件"; rm -f "$PID_FILE"; return 0
  fi
  if ! tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- "--port ${MY_PORT}\b"; then
    _guard_log "⛔ $p 的命令行里没有 --port ${MY_PORT} ⇒ 不是本会话的服务，拒绝杀"
    return 1
  fi
  _guard_log "停本会话服务：kill -TERM -$p"
  kill -TERM -"$p" 2>/dev/null
  rm -f "$PID_FILE"
}

wait_released () {
  local i free
  for i in $(seq 1 120); do
    nc -z 127.0.0.1 "$MY_PORT" 2>/dev/null && { sleep 5; continue; }
    free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    [ "${free:-0}" -ge "$VRAM_MIN_GIB" ] && { _guard_log "已释放（min free ${free} GiB，用了 $((i*5))s）"; return 0; }
    sleep 5
  done
  _guard_log "⚠️ 释放等待超时"; return 1
}
