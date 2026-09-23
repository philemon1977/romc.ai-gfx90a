#!/bin/bash
# 等并发会话的臂结束（容器消失 + 显存释放且连续确认）后，自动跑 GLM-5.3 验收
set -u
R=/home/qiba/ROCm.AI/quark-int8
LOG=$R/logs/glm53_watch_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "=== 开始等待 GPU（并发会话 dsv41-ct-int4）$(date +%T) ==="
ok=0
for i in $(seq 1 1080); do
  c=$(docker ps -q -f name=dsv41-ct-int4 | wc -l)
  used=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{s+=$NF} END{printf "%d", s/1073741824}')
  if [ "$c" = "0" ] && [ "${used:-999}" -lt 40 ]; then
    ok=$((ok+1)); echo "  空闲确认 $ok/3 (容器=$c 总显存=${used}GiB) $(date +%T)"
    [ "$ok" -ge 3 ] && break
  else
    [ "$ok" != "0" ] && echo "  又被占用/重启，计数归零 $(date +%T)"
    ok=0
  fi
  sleep 20
done
if [ "$ok" -lt 3 ]; then echo "❌ 等待超时（6 小时）"; exit 1; fi
echo "=== GPU 空闲，开始 GLM-5.3 验收 $(date +%T) ==="
bash /tmp/glm_final.sh
echo "WATCH_DONE $(date +%T)"
