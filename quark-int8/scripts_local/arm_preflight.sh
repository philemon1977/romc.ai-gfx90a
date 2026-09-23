#!/bin/bash
# 起臂前置检查（机制化，替代"靠记性"）。历次事故：
#   1) 05:07 一个"静默 5 分钟后自动 SIGABRT"的抓栈探针没有 PID 文件、起臂前无人检查，
#      按时打进了刚起的新臂 ⇒ 表现为"启动失败"。
#   2) 06:29 上一轮臂脚本（名字没进清理名单）收尾时用**共享 PID 文件** + `docker rm -f`
#      把新臂的容器与进程一起打掉 ⇒ 表现为"容器静默消失"。
#   3) 06:33 本清理只排除自己 `$$`，结果把**父脚本**也杀了 ⇒ 表现为"臂根本没起"。
# 用法：任何起臂脚本开头 `bash arm_preflight.sh`（可传自己的 PID 作为额外豁免）
set -u
echo "== 起臂前置检查 $(date +%T) =="
D="$(cd "$(dirname "$0")" && pwd)"
# 祖先链豁免：自己 + 父 + 祖父 … 直到 1（否则会杀掉调用自己的那个脚本）
EXEMPT=" $$ ${1:-}"
p=$$
while [ "$p" -gt 1 ] 2>/dev/null; do
  p=$(awk '{print $4}' /proc/$p/stat 2>/dev/null) || break
  [ -z "$p" ] && break
  EXEMPT="$EXEMPT $p"
done
echo "  豁免 PID:$EXEMPT"
for q in $(ps -eo pid,args | grep -F "$D/" | grep -v grep | awk '{print $1}'); do
  case " $EXEMPT " in *" $q "*) continue;; esac
  kill -TERM "$q" 2>/dev/null && echo "  清理遗留脚本 pid=$q"
done
sleep 5
PIDF=/home/qiba/ai/logs/dsv41ctint4-8119.pid
# 陈旧 PID 文件保护：>6 小时前写的一律不用（避免误杀别人的进程组）
if [ -f "$PIDF" ] && [ -n "$(find "$PIDF" -mmin -360 2>/dev/null)" ]; then
  kill -TERM -"$(cat $PIDF)" 2>/dev/null && echo "  已停上一轮服务"; sleep 8
else
  [ -f "$PIDF" ] && echo "  PID 文件陈旧（>6h），跳过停服以免误杀"
fi
docker rm -f dsv41-ct-int4 >/dev/null 2>&1 && echo "  已移除旧容器"
for i in $(seq 1 30); do
  U=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{s+=$NF} END{print s+0}')
  [ "$U" -lt 5368709120 ] && { echo "  显存已释放（合计 $((U/1073741824)) GiB 在用）"; break; }
  echo "  等待显存释放… 当前 $((U/1073741824)) GiB"; sleep 10
done
# ★ 他人占用检查：等不到就中止，绝不抢卡（本机铁律）
U=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{s+=$NF} END{print s+0}')
if [ "$U" -ge 5368709120 ]; then
  echo "❌ 显存仍被占用（合计 $((U/1073741824)) GiB 在用）且不是本会话的容器 —— 等待他人释放后再跑，不抢卡"
  exit 3
fi
echo "== 前置检查完成 =="