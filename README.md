# ROCm.AI Docker 环境栈

针对本机 **8× AMD Instinct MI250 (gfx90a / MI200)** 部署的 [AMD ROCm.AI](https://www.amd.com/en/products/software/rocm/rocm-ai.html) 官方容器环境集合。

ROCm.AI 平台由三部分组成：**AMD Skills**（面向 Cursor/Claude/Codex 等智能体编码工具的 ROCm 原生技能）、**Hyperloom**（AI 辅助调优层）、**ROCm Core SDK**（运行时/库/编译器/框架集成）。落到 Docker 上，就是 AMD 官方发布的 `rocm/*` 框架镜像系列（[ROCm AI Developer Hub](https://www.amd.com/zh-cn/developer/resources/rocm-hub/ai-cloud-development.html)）。

## 环境矩阵（见 `.env`）

| 环境 | 镜像 | 说明 |
|---|---|---|
| PyTorch | `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1` | ROCm 7.2.4 + PyTorch 2.9.1 |
| TensorFlow | `rocm/tensorflow:rocm7.14.1-ubuntu24.04-py3.12-tf2.20` | ROCm 7.14 + TF 2.20 |
| JAX | `rocm/jax:rocm7.2.4-jax0.8.2-py3.12` | ROCm 7.2 + JAX 0.8.2 |
| vLLM (LLM 推理) | `rocm-ai/vllm:0.28.0-rocm7.2.4`（**本地构建**） | base=`rocm/pytorch:rocm7.2.4`（本地已有）+ 复用 `/home/qiba/ai/envs/vllm_0.28.0_rocm72` conda 环境（vLLM 0.28.0 + torch 2.12 rocm7.2, gfx90a），补装 tqdm/gguf/filelock。见 `docker/Dockerfile.vllm-local`。**免去 ~20GB 官方 rocm/vllm 镜像**（其 6.4.1 线 vLLM 仅 0.10.1，更旧） |

## 验证状态（全部 PASS，容器内 GPU 计算通过）

| 环境 | 版本 | 结果 |
|---|---|---|
| PyTorch | 2.9.1+rocm7.2.4 | ✅ 8×MI250 gfx90a，4096³ fp16 matmul |
| TensorFlow | 2.20.0-dev0 (ROCm 7.14) | ✅ 8×MI250，2048³ matmul |
| JAX | 0.8.2 (ROCm 7.2) | ✅ 8×MI250，2048³ matmul |
| vLLM | 0.28.0 (ROCm 7.2) | ✅ Qwen2.5-0.5B 端到端生成 |

> 注：GPU0 上有他人进程（约占用 53GiB），测试时请用 `HIP_VISIBLE_DEVICES=1..7` 选择空闲卡。

> MI250 的 gfx90a 在 ROCm 7.x 系统需求中仍受支持（[参考](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)）。
> 若新镜像中 gfx90a kernel 缺失，回退方案见 `.env` 中的注释（ROCm 6.4.4 系列）。

## 下载加速经验（本机网络）

1. 直连 `hub.docker.com` / `hf-mirror.com` 被 SSL 截断；daemon 配的 `daocloud/1ms/xuanyuan` mirror 会 500/挂起。
   **有效方案**：显式加代理前缀拉取再重命名：
   `docker pull dockerproxy.net/rocm/<img>:<tag> && docker tag dockerproxy.net/... rocm/...`
2. 模型用 ModelScope：`https://modelscope.cn/models/<org>/<model>/resolve/master/<file>`。
3. PyPI 用清华源（`-i https://pypi.tuna.tsinghua.edu.cn/simple`）。
4. 各 `rocm/*` 镜像大量层共享，第二个镜像起下载量明显减少；能用本地已有资产（conda env、已拉镜像）就不要拉新镜像。
5. `docker tag` 前记得确认 tag 列表来源：`https://dockerproxy.net/v2/rocm/<img>/tags/list`。

## 使用

```bash
# 交互式进入某个环境（工作区挂载在容器 /workspace）
docker-compose run --rm pytorch
docker-compose run --rm tensorflow
docker-compose run --rm jax
docker-compose run --rm vllm

# 只用第 0、1 块 GPU
docker-compose run --rm -e HIP_VISIBLE_DEVICES=0,1 pytorch

# 冒烟测试（GPU 识别 + matmul）
./scripts/smoke.sh            # 全部
./scripts/smoke.sh pytorch    # 单个

# vLLM 起 OpenAI 兼容服务（models/ 下已备好 tiny Qwen2.5-0.5B）
docker-compose up -d vllm          # 默认命令即 vllm serve /models
# 或自定义:
docker-compose run --rm -e HIP_VISIBLE_DEVICES=1 vllm \
  vllm serve /models --served-model-name tiny-qwen --max-model-len 4096

# （重新）构建本地 vLLM 镜像
docker build -f docker/Dockerfile.vllm-local -t rocm-ai/vllm:0.28.0-rocm7.2.4 .
```

## 主机前提（已满足）

- amdgpu 驱动 ≥ 6.x（本机 6.16.13）
- `/dev/kfd`、`/dev/dri` 存在，用户在 `video`/`render` 组
- Docker 配置了国内 registry mirror（`/etc/docker/daemon.json`，直连 hub.docker.com 被墙）

## 目录

本项目工作区（代码 / 文档 / 配置 / 脚本 / 测试 / 补丁 / 知识库 / 报告与实测数据 / 产出）的索引：

- `docker-compose.yml` — 四个环境服务，共享 GPU/挂载配置
- `.env.example` — 镜像 tag 矩阵与回退（本机 `.env` 不入库，`cp .env.example .env`）
- `docker/Dockerfile.vllm-local` — 本地 vLLM 镜像构建（复用既有资产，省 ~20 GB 拉取）
- `tests/` — 各环境冒烟测试脚本；`scripts/smoke.sh`、`scripts/pull_all.sh`、`scripts/pull_rest.sh`
- `scripts/bootstrap.sh` — 还原不入库的第三方代码 / 依赖 / 模型（按 `DEPENDENCIES.md` 的锁定版本）
- `scripts/build_fp8_emulation_overlay.sh` — 由已提交的 `.patch` 重建 gfx90a FP8-emulation overlay 树
- `scripts/fetch_models.sh` — 从 ModelScope 拉 tiny-qwen 冒烟模型
- `scripts/seed_recipe_kb.py`、`scripts/verify_recipe_kb.py`、`scripts/build_skill_data.py` — 配方知识库灌数与校验
- `scripts/note_*.py` — 现场取证脚本（agent lane / emulation boot / probe crosscheck）
- `hyperloom/patches/` — **gfx90a FP8→BF16 dequant emulation 补丁**（`*.patch` + `bin/vllm-emulation` 启动器）
- `hyperloom/patches-local/` — ROCm 设备掩码 reconcile 补丁
- `hyperloom/kernels/` — 自写 gfx90a flash-decode / split-KV kernel、补丁与基准
- `hyperloom/kb/` — MI250x 配方知识库（按 模型 × 框架 × 量化 归档，含历史版本）
- `hyperloom/presets/`、`hyperloom/scripts-local/`、`hyperloom/session-priors/` — 本机 preset 与运行脚本
- `hyperloom/reports/` — **实测报告与产出索引**（入口：`hyperloom/reports/README.md`）
- `local-skills/mi250x-recipe-ops/` — 本机自建 Skill 及其数据

第三方代码（**不入库**，由 `scripts/bootstrap.sh` 还原，见 [`DEPENDENCIES.md`](DEPENDENCIES.md)）：

- `amd-skills/` — 官方 [amd/skills](https://github.com/amd/skills) 克隆（锁定 commit `6916fb3`）；技能已软链接至 `~/.dsh/skills/`、`~/.claude/skills/`、`~/.agents/skills/`
- `hyperloom/hyperloom`、`hyperloom/kernelforge`、`hyperloom/bin` — [AMD-AGI/Hyperloom](https://github.com/AMD-AGI/Hyperloom) 1.1.0（MIT）的 pip 安装副本

## 仓库范围：91 GB 工作区 vs 仓库内容

工作区总占用约 **91 GB**，其中绝大部分是**可再生或纯重复**的资产，全部排除在 git 之外：

| 排除项 | 体积 | 原因 |
|---|---|---|
| `envs/`、`hyperloom/envs/` | ~26 GB | Python 虚拟环境，`bootstrap.sh` 重建 |
| `hyperloom/session/` | ~29 GB | optimizer 运行现场：每个候选一棵 vLLM worktree + AOT 编译缓存 |
| `hyperloom/.tmp/`、`.tmp/`、`hyperloom/.cache/` | ~34 GB | 临时实验目录与上游仓库缓存 |
| `hyperloom/patches/*/vllm/**` | ~1.4 GB | 已构建 overlay 树；其中 `.so` 经 **sha256 与上游 wheel 逐字节相同**，Python 改动全在 `.patch` 里 |
| `models/` | ~1 GB | 第三方模型权重，`fetch_models.sh` 拉取 |
| `hyperloom/hyperloom/` 等 | ~40 MB | 第三方代码副本，见 `DEPENDENCIES.md` |

**入库的二进制**走 Git LFS（合计 ~797 MB，只收"不可再生"的实测证据）：
`a8w8-fix/module_gemm_a8w8.gfx90a-{fixed,orig}.so`（AITER `M<=64` 边界修复产物 + 修复前基线）
与 `profiling/benchmark_vllm_20260915_051456/torch_trace/*.pt.trace.json.gz`（decode 归因的原始 trace）。
完整逐项理由见 [`DEPENDENCIES.md`](DEPENDENCIES.md) 第 6 节。

## 克隆与还原

```bash
# 轻量克隆（推荐）：跳过 797 MB LFS 下载
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/philemon1977/romc.ai-gfx90a.git ROCm.AI
cd ROCm.AI

# 需要实测数据/修复产物时再拉 LFS
export PATH="$PWD/.tools/bin:$PATH"    # git-lfs 由 bootstrap 装到工作区（本机无 sudo）
scripts/bootstrap.sh lfs && git lfs pull

scripts/bootstrap.sh                   # 交互式还原第三方代码 / 依赖 / 模型
cp .env.example .env                   # 本机镜像 tag 配置
./scripts/smoke.sh pytorch             # 验证 GPU 计算通路
```

`hyperloom/reports/models/qwen38-27b-w8a8-dense/profiling/` 下的 trace 可用
[Magpie TraceLens](https://github.com/AMD-AGI/Magpie) 后处理成 prefill/decode 与 roofline 报告。
