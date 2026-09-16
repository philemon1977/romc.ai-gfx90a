#!/usr/bin/env bash
# =============================================================================
# ROCm.AI @ 8x MI250 (gfx90a) —— 工作区还原
#
# 仓库刻意不含第三方代码、虚拟环境、模型权重与 optimizer 运行现场
# （理由与逐项清单见 DEPENDENCIES.md）。本脚本按其中的锁定版本还原它们。
#
# 用法：
#   scripts/bootstrap.sh              # 交互式，逐项确认
#   scripts/bootstrap.sh --all        # 全部还原（含 ~20 GB 镜像，慎用）
#   scripts/bootstrap.sh lfs skills   # 只做指定步骤
#
# 步骤： lfs | skills | hyperloom | envs | images | models | smoke
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 本机经验（README「下载加速经验」）：直连 PyPI/Docker Hub 不稳，默认走国内源。
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
DOCKER_PYTHON="${DOCKER_PYTHON:-python3}"

# 锁定版本 —— 与 DEPENDENCIES.md 保持一致
AMD_SKILLS_REPO="https://github.com/amd/skills.git"
AMD_SKILLS_COMMIT="6916fb371d1cba40757b223cf16b4cd4912b202f"
HYPERLOOM_VERSION="1.1.0"
GIT_LFS_VERSION="3.8.0"

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mxx %s\033[0m\n' "$*" >&2; exit 1; }

confirm() {
  [[ "$NONINTERACTIVE" == "1" ]] && return 0
  read -r -p "   → $1 [y/N] " ans
  [[ "${ans:-N}" =~ ^[Yy] ]]
}

# ---------------------------------------------------------------- lfs --------
step_lfs() {
  say "git-lfs → ./.tools/bin（本机无 sudo，装在工作区内）"
  # .tools/ 是 gitignore 的，全新克隆后并不存在；PATH 必须先于任何 git-lfs 调用设好。
  if ! command -v git-lfs >/dev/null 2>&1 && [[ -x "$ROOT/.tools/bin/git-lfs" ]]; then
    export PATH="$ROOT/.tools/bin:$PATH"; hash -r
  fi
  if ! command -v git-lfs >/dev/null 2>&1; then
    confirm "下载 git-lfs v${GIT_LFS_VERSION} 静态二进制到 .tools/bin？" || \
      { warn "跳过；提交/取回二进制前必须装好"; return 0; }
    local tmp; tmp="$(mktemp -d)"
    curl -fsSL -o "$tmp/lfs.tgz" \
      "https://github.com/git-lfs/git-lfs/releases/download/v${GIT_LFS_VERSION}/git-lfs-linux-amd64-v${GIT_LFS_VERSION}.tar.gz" \
      || die "git-lfs 下载失败（网络受限时可手动放到 .tools/bin/git-lfs）"
    tar -C "$tmp" -xzf "$tmp/lfs.tgz"
    mkdir -p "$ROOT/.tools/bin"
    install -m 0755 "$tmp/git-lfs-${GIT_LFS_VERSION}/git-lfs" "$ROOT/.tools/bin/git-lfs"
    rm -rf "$tmp"
    export PATH="$ROOT/.tools/bin:$PATH"; hash -r
  fi
  echo "   git-lfs: $(git-lfs version)"
  # 全新克隆没有 filter.lfs.* 配置；缺了它 checkout 只会拿到 133 字节的指针文件。
  if git lfs install --local >/dev/null 2>&1; then
    echo "   本克隆已启用 LFS 过滤器：$(git config --local filter.lfs.process)"
  else
    warn "git lfs install --local 失败，手动执行一次"
  fi
  warn "新开的 shell 里需要：export PATH=\"\$PWD/.tools/bin:\$PATH\""
}

# -------------------------------------------------------------- skills -------
step_skills() {
  say "AMD Skills（第三方 clone，锁定 commit）"
  if [[ -d amd-skills/.git ]]; then
    echo "   已存在，跳过。当前：$(git -C amd-skills rev-parse HEAD 2>/dev/null || echo unknown)"
    return 0
  fi
  confirm "clone ${AMD_SKILLS_REPO##*/} @ ${AMD_SKILLS_COMMIT:0:8} 到 ./amd-skills/？" || return 0
  git init -q amd-skills
  git -C amd-skills remote add origin "$AMD_SKILLS_REPO"
  # 单 commit 浅取：上游仓库对 ROCm 技能是整树分发，只固定到记录的那个 commit。
  git -C amd-skills fetch -q --depth 1 origin "$AMD_SKILLS_COMMIT" \
    || die "fetch 失败：检查网络或改用代理"
  git -C amd-skills checkout -q FETCH_HEAD
  echo "   OK: $(git -C amd-skills log -1 --format='%h %s')"
}

# ----------------------------------------------------------- hyperloom -------
step_hyperloom() {
  say "Hyperloom ${HYPERLOOM_VERSION}（MIT，PyPI: hyperloom-inference-optimizer）"
  if [[ -d hyperloom/hyperloom && -d hyperloom/kernelforge ]]; then
    echo "   已存在安装树 hyperloom/{hyperloom,kernelforge,bin}，跳过。"
    return 0
  fi
  confirm "pip install --prefix ./hyperloom 'hyperloom-inference-optimizer[runtime]==${HYPERLOOM_VERSION}'？" || return 0
  command -v "$DOCKER_PYTHON" >/dev/null || die "需要 python3"
  "$DOCKER_PYTHON" -m venv envs/hyperloom 2>/dev/null || true
  # --prefix 还原成仓库里原有的布局：hyperloom/{hyperloom,kernelforge,bin,*-1.1.0.dist-info}
  PIP_INDEX_URL="$PIP_INDEX_URL" "$DOCKER_PYTHON" -m pip install \
    --prefix "$ROOT/hyperloom" --ignore-installed \
    "hyperloom-inference-optimizer[runtime]==${HYPERLOOM_VERSION}" \
    || die "pip 安装失败；上游源码：https://github.com/AMD-AGI/Hyperloom/tree/v${HYPERLOOM_VERSION}"
  echo "   OK. 入口：hyperloom/bin/hyperloom"
}

# ---------------------------------------------------------------- envs -------
step_envs() {
  say "Python 环境（envs/，~12 GB，不入库）"
  echo "   本仓库的 vLLM 环境是'从 conda 环境复制 + 补装依赖'得来的，见 docker/Dockerfile.vllm-local。"
  echo "   重建方式二选一："
  echo "     a) docker-compose build vllm   （走 images 步骤）"
  echo "     b) python3 -m venv envs/vllm-0.28.0-rocm7.2.4 && PIP_INDEX_URL=$PIP_INDEX_URL \\"
  echo "          envs/vllm-0.28.0-rocm7.2.4/bin/pip install 'vllm==0.28.0'   # 需 ROCm 7.2 wheel 源"
}

# --------------------------------------------------------------- images ------
step_images() {
  say "ROCm 容器镜像（~20 GB 级，走国内代理）"
  command -v docker >/dev/null || die "未检测到 docker"
  local envfile=".env"; [[ -f "$envfile" ]] || envfile=".env.example"
  local imgs
  imgs="$(set -a; . "./$envfile"; set +a; printf '%s\n' "${PYTORCH_IMAGE:-}" "${TF_IMAGE:-}" "${JAX_IMAGE:-}" | grep -v '^$' || true)"
  echo "   镜像清单来自 $envfile："
  printf '     %s\n' $imgs
  confirm "docker pull 上述官方镜像？" || return 0
  local img
  for img in $imgs; do
    if docker image inspect "$img" >/dev/null 2>&1; then
      echo "   已有：$img"
    else
      # README 记录：直连 hub.docker.com 被 SSL 截断，需要显式加代理前缀再重命名。
      warn "拉取 $img（直连失败时：docker pull dockerproxy.net/$img && docker tag dockerproxy.net/$img $img）"
      docker pull "$img" || warn "直连失败，请手动走 dockerproxy.net 前缀"
    fi
  done
  confirm "构建本地 vLLM 镜像（docker/Dockerfile.vllm-local）？" && \
    docker-compose build vllm || warn "跳过本地 vLLM 镜像构建"
}

# --------------------------------------------------------------- models ------
step_models() {
  say "冒烟模型（models/，不入库）"
  if [[ -f models/model.safetensors ]]; then echo "   已存在，跳过。"; return 0; fi
  [[ -x scripts/fetch_models.sh ]] || chmod +x scripts/fetch_models.sh
  confirm "下载 tiny Qwen2.5-0.5B（~1 GB，ModelScope）到 ./models/？" && scripts/fetch_models.sh || warn "跳过模型下载"
}

# ---------------------------------------------------------------- smoke ------
step_smoke() {
  say "冒烟测试"
  [[ -x scripts/smoke.sh ]] || chmod +x scripts/smoke.sh
  confirm "运行 scripts/smoke.sh all（需 GPU 空闲，注意 GPU0 上可能有他人进程）？" && \
    ./scripts/smoke.sh all || warn "跳过冒烟测试"
}

# ----------------------------------------------------------------- main ------
STEPS=("$@")
NONINTERACTIVE=0
if [[ "${STEPS[0]:-}" == "--all" ]]; then NONINTERACTIVE=1; STEPS=(lfs skills hyperloom envs images models smoke); fi
[[ ${#STEPS[@]} -eq 0 ]] && STEPS=(lfs skills hyperloom envs images models smoke)

echo "工作区：$ROOT"
[[ -f .env ]] || { say "缺少 .env（本机配置，不入库）"; cp -v .env.example .env && warn "已从 .env.example 生成，按需修改"; }

for s in "${STEPS[@]}"; do
  case "$s" in
    lfs)        step_lfs ;;
    skills)     step_skills ;;
    hyperloom)  step_hyperloom ;;
    envs)       step_envs ;;
    images)     step_images ;;
    models)     step_models ;;
    smoke)      step_smoke ;;
    *) die "未知步骤：$s（可用：lfs skills hyperloom envs images models smoke | --all）" ;;
  esac
done

say "完成"
cat <<'EOF'
下一步：
  docker-compose run --rm pytorch      # 进入某个环境
  ./scripts/smoke.sh vllm              # GPU 冒烟
  hyperloom --help                     # 优化器（需 hyperloom 步骤 + envs）
EOF
