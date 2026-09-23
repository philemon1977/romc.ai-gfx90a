#!/bin/bash
# v3：等全量转换结束 → 就地 requant engram → 结构/体积校验。
# 相对 v2 的改动：目录换成 -opt-shbf16；转换日志路径由参数传入（不再读 /tmp 里的旧文件）。
#
# 两处不变的关键点：
#   1) requant 前先等一个 ≥FREE_MIN_GIB 的空闲显存窗口，等不到就退回 --device cpu（不抢卡）。
#   2) requant 必须**就地**（--src 与 --dst 同为新目录）：它会把该分片「其余张量原样搬运」，
#      而那些是从 --src 读的；分片 47/48 里 engram.wkv 已被转换器改成 bf16。
set -u
NEW=${NEW:-/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4-opt-shbf16}
CLOG=${CLOG:?需要传入转换日志路径}
REPO=/home/qiba/ROCm.AI/quark-int8
FREE_MIN_GIB=${FREE_MIN_GIB:-32}
WAIT_MAX_S=${WAIT_MAX_S:-1200}

free_max() {
  local m=0 v
  for d in 0 1 2 3 4 5 6 7; do
    v=$(rocm-smi --showmeminfo vram -d "$d" 2>/dev/null | awk '/Used Memory/{print $NF}')
    if [ -n "${v:-}" ]; then v=$(( (68702699520 - v) / 1048576 )); else v=0; fi
    [ "$v" -gt "$m" ] && m=$v
  done
  echo "$m"
}

echo "=== $(date +%T) 等转换结束（clog=$CLOG）==="
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

echo "=== $(date +%T) 体积（预期 ≈399 GiB：397.30 + 共享专家 40×3×[2304,5120] 由 int4 升 bf16）==="
du -sh "$NEW"
ls -la "$NEW"/model-0004[78]-of-00048.safetensors
echo "=== $(date +%T) 完毕 ==="
