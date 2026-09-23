#!/bin/bash
# 收尾守卫：只有"本脚本启动之后才创建"的容器才允许被本脚本停掉/删除。
# 起因（2026-09-20 06:29）：上一轮臂脚本的收尾把新臂的容器 rm 掉，容器静默消失。
# 用法：source 本文件；收尾时 if owns_container; then ... fi
DSV41_ARM_START_EPOCH=$(date +%s)
owns_container() {
  local started
  started=$(docker inspect -f "{{.State.StartedAt}}" dsv41-ct-int4 2>/dev/null) || return 1
  [ -n "$started" ] || return 1
  local sepoch
  sepoch=$(date -d "$started" +%s 2>/dev/null) || return 1
  [ "$sepoch" -ge "$DSV41_ARM_START_EPOCH" ]
}
