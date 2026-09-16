#!/usr/bin/env bash
# =============================================================================
# 重建 gfx90a FP8->BF16 dequant-emulation 的 vLLM overlay 树
#
# 仓库里只保存 `fp8-w8a8-emulation-gfx90a.patch`（Python 侧 4 个文件的全部改动）
# 和 `bin/vllm-emulation` 启动器；overlay 树里的 .so 已用 sha256 验证与上游 wheel
# 逐字节相同，属纯重复，故不入库（见 DEPENDENCIES.md 第 5 节）。
#
# 本脚本把已安装的 vLLM 复制成一棵可 PYTHONPATH 前置加载的 overlay，并打上补丁。
#
# 用法：
#   scripts/build_fp8_emulation_overlay.sh [SRC_VLLM_DIR] [DST_OVERLAY_ROOT]
#
#     SRC_VLLM_DIR  默认 envs/vllm-0.28.0-rocm7.2.4/lib/python3.12/site-packages
#                   （容器内跑时为 /opt/envs/vllm/lib/python3.12/site-packages）
#     DST_OVERLAY_ROOT 默认 hyperloom/patches/fp8-w8a8-emulation-gfx90a
#
# 打完补丁后用 bin/vllm-emulation 启动（VLLM_BIN 指向它即可被 preset 使用）：
#   hyperloom ... --extra-env VLLM_BIN=$PWD/hyperloom/patches/fp8-w8a8-emulation-gfx90a/bin/vllm-emulation
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SRC="${1:-$ROOT/envs/vllm-0.28.0-rocm7.2.4/lib/python3.12/site-packages}"
DST_ROOT="${2:-$ROOT/hyperloom/patches/fp8-w8a8-emulation-gfx90a}"
PATCH="$DST_ROOT/fp8-w8a8-emulation-gfx90a.patch"

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mxx %s\033[0m\n' "$*" >&2; exit 1; }

[[ -d "$SRC/vllm" ]] || die "找不到源 vLLM 树：$SRC/vllm（先跑 scripts/bootstrap.sh envs，或传入容器内的 site-packages 路径）"
[[ -f "$PATCH"  ]] || die "找不到补丁：$PATCH"

SRC_PKG="$SRC/vllm"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

say "1/4 复制 vLLM 包（不含 __pycache__）"
rsync -a --exclude='__pycache__' --exclude='*.pyc' "$SRC_PKG" "$BUILD/installed_vllm/"
du -sh "$BUILD/installed_vllm" | sed 's/^/   /'

say "2/4 打补丁（patch -p0，源目录名与补丁头一致）"
cp -a "$BUILD/installed_vllm" "$BUILD/overlay_vllm"
# 补丁是 `diff -ruN -x __pycache__ installed_vllm/... overlay_vllm/...` 生成的，
# 前缀即目录名，因此在 $BUILD 下直接 -p0 应用。
( cd "$BUILD" && patch -p0 --no-backup-if-mismatch -i "$PATCH" ) \
  || die "补丁未能干净应用：vLLM 版本可能与生成补丁时的 0.28.0+rocm723 不一致"

say "3/4 校验改动文件"
changed=0
while IFS= read -r rel; do
  if ! cmp -s "$BUILD/installed_vllm/$rel" "$BUILD/overlay_vllm/$rel"; then
    echo "   * $rel"
    changed=$((changed + 1))
  fi
done < <(grep -oE '^diff -ruN [^ ]+' "$PATCH" | awk '{print $NF}' | sed 's|^overlay_vllm/||')
echo "   共 $changed 个文件与源树不同"
[[ $changed -gt 0 ]] || die "补丁应用后没有任何文件变化，中止"

say "4/4 落盘到 $DST_ROOT/vllm"
mkdir -p "$DST_ROOT"
rm -rf "$DST_ROOT/vllm.tmp"
cp -a "$BUILD/overlay_vllm" "$DST_ROOT/vllm.tmp"
rm -rf "$DST_ROOT/vllm.old"
[[ -d "$DST_ROOT/vllm" ]] && mv "$DST_ROOT/vllm" "$DST_ROOT/vllm.old"
mv "$DST_ROOT/vllm.tmp" "$DST_ROOT/vllm"
[[ -x "$DST_ROOT/bin/vllm-emulation" ]] || chmod +x "$DST_ROOT/bin/vllm-emulation" 2>/dev/null || true

cat <<EOF

完成。overlay 树：$DST_ROOT/vllm
启动器：$DST_ROOT/bin/vllm-emulation  （它把该目录放到 PYTHONPATH 最前，并固定
        VLLM_ROCM_USE_AITER=0 / 独立 VLLM_CACHE_ROOT —— 原因见脚本内注释）

冒烟验证（gfx90a 无 FP8 矩阵核心，走 BF16 dequant 路径）：
  $DST_ROOT/bin/vllm-emulation --version
  $DST_ROOT/bin/vllm-emulation serve <fp8-model> --gpu-memory-utilization 0.5
EOF
