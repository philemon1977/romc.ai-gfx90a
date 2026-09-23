#!/usr/bin/env bash
# kioxia × AIS 连通性与带宽测试（DSV4.1 启动加速的硬件前提验证）
#
# 步骤：
#   1. 取 hipfile-dev deb（GitHub nightly，与主包同版号）→ dpkg -x 合并进 staging，
#      拿到 hipfile.h（红线：只 dpkg -x，绝不 apt install —— 见部署手册 N5）
#   2. hipcc 编译官方示例 aiscp（file → GPU → file，读路径走 hsa_amd_ais_file_read）
#   3. 在 kioxia 上造 1 GiB 测试文件
#   4. aiscp 吞吐 vs dd(iflag=direct) 吞吐，判定 AIS 是否在 kioxia 上工作
#
# 用法：bash prep_ais_kioxia.sh          # 全跑（含 I/O，等机器空闲再跑）
#       SKIP_IO=1 bash prep_ais_kioxia.sh  # 只装工具链
set -uo pipefail

AI=/home/qiba/ai
S=$AI/cache/hipfile-stage/opt/rocm-7.2.0
STAGE_ROOT=$AI/cache/hipfile-stage
BASE=https://github.com/ROCm/hipFile/releases/download/nightly
DEV_DEB=hipfile-dev_0.2.0.70200-nightly.9999.24.04_amd64.deb
TMP=$(mktemp -d /tmp/ais.XXXXXX)
KIOXIA_TEST=/mnt/kioxia-cm6-3t8/ai/tmp/ais_kioxia_t1g.bin
OUT=/tmp/ais_out_$$.bin

echo "== 1) hipfile-dev 头文件 =="
if [ -f "$S/include/hipfile.h" ]; then
    echo "  已有 $S/include/hipfile.h"
else
    timeout 60 curl -sL -o "$TMP/$DEV_DEB" "$BASE/$DEV_DEB" || { echo "❌ 下载失败"; exit 1; }
    dpkg -x "$TMP/$DEV_DEB" "$TMP/dev"
    HDR=$(find "$TMP/dev" -name hipfile.h | head -1)
    [ -n "$HDR" ] || { echo "❌ deb 里没有 hipfile.h"; exit 1; }
    mkdir -p "$S/include"
    cp -n "$TMP/dev"/opt/rocm*/include/* "$S/include/" 2>/dev/null || cp "$HDR" "$S/include/"
    echo "  已安装 $(find "$S/include" -name "*.h*" | wc -l) 个头文件到 staging"
fi

echo "== 2) 编译 aiscp =="
EX=$S/share/doc/hipfile/examples/aiscp
if [ -x "$TMP/aiscp" ] || hipcc "$EX/aiscp.cpp" -I"$S/include" -I/opt/rocm-7.2.4/include \
     -L"$S/lib" -lhipfile -o "$TMP/aiscp" 2>"$TMP/build.log"; then
    [ -x "$TMP/aiscp" ] || cp aiscp "$TMP/aiscp" 2>/dev/null
    echo "  编译 OK: $TMP/aiscp"
else
    echo "❌ 编译失败:"; tail -5 "$TMP/build.log"; exit 1
fi

[ "${SKIP_IO:-0}" = "1" ] && { echo "(SKIP_IO=1，I/O 测试未跑)"; exit 0; }

echo "== 3) kioxia 1 GiB 测试文件 =="
mkdir -p "$(dirname "$KIOXIA_TEST")"
[ -f "$KIOXIA_TEST" ] || dd if=/dev/zero of="$KIOXIA_TEST" bs=1M count=1024 status=none

echo "== 4) dd 直读基线 =="
T0=$(date +%s%N)
dd if="$KIOXIA_TEST" of=/dev/null bs=1M count=1024 iflag=direct status=none
DD_MS=$(( ($(date +%s%N) - T0) / 1000000 ))
echo "  dd direct: ${DD_MS} ms  ($(( 1024 * 1000 / DD_MS )) MB/s)"

echo "== 5) aiscp（file→GPU→tmpfs）=="
T0=$(date +%s%N)
LD_LIBRARY_PATH=$S/lib HIP_VISIBLE_DEVICES=7 "$TMP/aiscp" "$KIOXIA_TEST" "$OUT" \
  || { echo "❌ aiscp 失败（AIS 路径在 kioxia 上不通？看上面报错）"; exit 1; }
AIS_MS=$(( ($(date +%s%N) - T0) / 1000000 ))
echo "  aiscp: ${AIS_MS} ms  ($(( 1024 * 1000 / AIS_MS )) MB/s，含 GPU→tmpfs 写回)"

echo
if [ "$AIS_MS" -lt $(( DD_MS * 3 )) ]; then
    echo "✅ kioxia AIS 路径可用（带宽同量级，P2P 读工作正常）"
    echo "   下一步：容器路线接 fastsafetensors —— 需把 staging 的 lib+include 以"
    echo "   glibc 2.35 兼容方式进容器（shim .so 只要求 GLIBC_2.34 ✓，hipfile .so 待查）"
else
    echo "❌ AIS 明显慢于直读/失败 ⇒ kioxia 不在 P2P 快速路径上（拓扑/IOMMU 限制）"
    echo "   退回 sharded_state 预切分路线"
fi
rm -f "$OUT" "$KIOXIA_TEST"
