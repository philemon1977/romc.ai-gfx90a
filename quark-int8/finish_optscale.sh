#!/bin/bash
# v2：等全量转换结束 → 就地做 engram fp8→int4 重写 → 结构/体积校验。
#
# 两处加固（对应 CLAUDE.md §1「机制化的做法不是自觉，而是脚本」）：
#   1) requant 原本直接走 GPU 而**没有同租户显存门**：若此刻别人起了任务就会抢卡。
#      现在先轮询等一个 ≥FREE_MIN_GIB 的空闲显存窗口，等不到就退回 --device cpu（完全不占卡）。
#   2) 就地调用（--src 与 --dst 都是新目录）：requant 会把该分片「其余张量原样搬运」，
#      而那是从 --src 读的；分片 47/48 里 engram.wkv 已被转换器改成 bf16，
#      若从源 fp8 目录读取就会拿回 fp8，造成同分片内格式不一致。
set -u
NEW=/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4-optscale
CLOG=$(cat /tmp/optscale_log_path)
REPO=/home/qiba/ROCm.AI/quark-int8
FREE_MIN_GIB=${FREE_MIN_GIB:-32}
WAIT_MAX_S=${WAIT_MAX_S:-1200}

free_max() {   # 八张卡里最大的空闲 MiB
  local m=0 v
  for d in 0 1 2 3 4 5 6 7; do
    v=$(rocm-smi --showmeminfo vram -d "$d" 2>/dev/null | awk '/Used Memory/{print $NF}')
    [ -n "${v:-}" ] && v=$(( (68702699520 - v) / 1048576 )) || v=0
    [ "$v" -gt "$m" ] && m=$v
  done
  echo "$m"
}

echo "=== $(date +%T) 等转换结束 ==="
while pgrep -f "convert_dsv41_ct_int4.py" >/dev/null; do sleep 30; done
echo "=== $(date +%T) 转换进程已退出 ==="
tail -6 "$CLOG"

echo "=== 守卫：48 个分片是否齐全 ==="
n=$(ls "$NEW"/model-000*-of-00048.safetensors 2>/dev/null | wc -l)
echo "分片数=$n"
if [ "$n" -ne 48 ]; then echo "❌ 分片不全，停止（不跑 requant）"; exit 1; fi
if ! grep -qa "DONE" "$CLOG"; then echo "❌ 日志未见 DONE，停止"; exit 1; fi

echo "=== $(date +%T) 等空闲显存窗口（需 ≥${FREE_MIN_GIB} GiB，最多 ${WAIT_MAX_S}s）==="
DEV=cpu
for i in $(seq 1 $((WAIT_MAX_S/20))); do
  m=$(free_max); echo "  t=$((i*20))s 最大空闲=${m} MiB"
  if [ "$m" -ge $((FREE_MIN_GIB*1024)) ]; then DEV=cuda; break; fi
  sleep 20
done
echo ">>> 选用设备：$DEV"

echo "=== $(date +%T) 就地 requant engram → int4（device=$DEV）==="
cd "$REPO"
DOCKER_DEV=()
[ "$DEV" = cuda ] && DOCKER_DEV=(--device /dev/kfd --device /dev/dri --group-add 44 --group-add 993)
docker run --rm --entrypoint bash "${DOCKER_DEV[@]}" --user 1000:1000 \
  -e HOME=/tmp -e USER=qiba \
  -v /home/qiba/ROCm.AI:/w \
  -v "$NEW":"$NEW" \
  vllm/vllm-openai-rocm:nightly \
  -c "python3 /w/quark-int8/requant_engram_int4.py --src $NEW --dst $NEW --device $DEV" 2>&1 | tail -25

echo "=== $(date +%T) 体积 ==="
du -sh "$NEW"
ls -la "$NEW"/model-0004[78]-of-00048.safetensors
echo "=== $(date +%T) 完毕 ==="
