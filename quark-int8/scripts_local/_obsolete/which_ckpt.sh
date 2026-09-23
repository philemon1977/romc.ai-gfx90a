#!/bin/bash
echo "== 04:00~06:59 起服日志的模型路径 =="
for L in $(ls -t /home/qiba/ai/logs/dsv41ctint4/server-8119-*.log | head -60); do
  hh=$(stat -c %y "$L" | cut -c12-13)
  case "$hh" in 04|05|06)
    p=$(grep -am1 "模型 /" "$L" | sed 's/.*模型 //')
    echo "$(stat -c %y "$L" | cut -c1-16)  $(basename "$L")  [$p]"
    ;;
  esac
done | head -14
echo
echo "== 各日志里出现的仓名统计 =="
grep -rhoE "DeepSeek-V4.1-Flash-CT-Int4-W4A16[a-z0-9-]*" /home/qiba/ROCm.AI/quark-int8/logs/*.log 2>/dev/null | sort | uniq -c | sort -rn | head -8
