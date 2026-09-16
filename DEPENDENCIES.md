# 依赖与锁定版本（DEPENDENCIES）

本仓库刻意**不包含**下列第三方资产与生成物：它们是官方分发的副本、或可由已提交的
补丁/脚本重新生成。此文件记录精确锁定版本，`scripts/bootstrap.sh` 据此还原。

工作区总占用约 **91 GB**；仓库提交的是其中的文档、代码、脚本、补丁、知识库与实测数据。

---

## 1. 宿主机基线

| 项目 | 值 | 校验命令 |
|---|---|---|
| GPU | 8 × AMD Instinct MI250 (gfx90a / MI200, CDNA2, 双 die) | `rocm-smi --showproductname` |
| OS | Ubuntu 24.04 | `lsb_release -a` |
| 内核 / amdgpu | 6.16.13 | `modinfo amdgpu \| grep version` |
| 容器运行时 | Docker 29（走国内镜像加速，见 README「下载加速经验」） | `docker version` |
| ROCm 主线 | 7.2.4（回退线 6.4.4，见 `.env.example` 注释） | — |

## 2. 容器镜像（`docker-compose.yml` + `.env.example`）

| 环境 | 镜像 |
|---|---|
| PyTorch | `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1` |
| TensorFlow | `rocm/tensorflow:rocm7.14.1-ubuntu24.04-py3.12-tf2.20` |
| JAX | `rocm/jax:rocm7.2.4-jax0.8.2-py3.12` |
| vLLM | `rocm-ai/vllm:0.28.0-rocm7.2.4`（**本地构建**，见 `docker/Dockerfile.vllm-local`） |

## 3. Python 侧版本（`envs/vllm-0.28.0-rocm7.2.4/`，不入库）

由本地构建的 vLLM 镜像 / conda 环境提供，实测版本：

| 包 | 版本 |
|---|---|
| vLLM | `0.28.0+rocm723` |
| PyTorch | `2.12.0+git6bbd260` (ROCm 7.2) |
| Triton | `3.7.1+gitf0b55c07` |
| flash-attn | `2.8.3` |
| aiter | AMD 版本随镜像（`site-packages/aiter`） |
| Python | 3.12 |

## 4. 第三方代码（不入库，`scripts/bootstrap.sh` 还原）

| 目录 | 上游 | 锁定 |
|---|---|---|
| `amd-skills/` | <https://github.com/amd/skills> | commit `6916fb371d1cba40757b223cf16b4cd4912b202f`（2026-09-09，"Use OIDC token federation (#209)"） |
| `hyperloom/hyperloom/`, `hyperloom/kernelforge/`, `hyperloom/bin/`, `hyperloom/*.dist-info/` | <https://github.com/AMD-AGI/Hyperloom>（MIT） | `hyperloom-inference_optimizer == 1.1.0` |
| `hyperloom/.cache/GEAK@…` | GEAK（kernel agent） | commit `c0c0e2aee5e2bec70583253058382523bdf7a3ab` |
| `hyperloom/.cache/InferenceX@…` | InferenceX（MI250x 基准脚本来源） | commit `3d5581562f643f9bdeb8410cd924e2c70906c966` |
| `hyperloom/.cache/TraceLens@…` | Magpie TraceLens（trace 后处理） | commit `a59a9c165bb64c7c416fd7cf79149803d552e43c` |
| `models/` | ModelScope `Qwen/Qwen2.5-0.5B-Instruct`（tiny 冒烟模型） | 见 `scripts/fetch_models.sh` |

> `hyperloom/.cache/*` 与 `hyperloom/session/**/wt*/` 里的 `site-packages/` 是 optimizer
> 为每个候选方案创建的 vLLM 工作副本，**体积 25 GB+ 且完全可再生**，故不入库。

## 5. 补丁与生成物：为什么只提交 `.patch`

`hyperloom/patches/fp8-w8a8-emulation-gfx90a/` 下有一棵 4079 文件 / 1.4 GB 的
`vllm/` overlay 树（供 `bin/vllm-emulation` 通过 `PYTHONPATH` 前置加载）。

其中 4 个 `.so` 已用 sha256 与 `envs/.../site-packages/vllm/` 逐一比对：
`_rocm_C.abi3.so` / `_C.abi3.so` / `_C_stable_libtorch.abi3.so` / `_moe_C_stable_libtorch.abi3.so`
**与上游 wheel 逐字节相同**，不含任何本项目改动；Python 侧的全部改动都在
`fp8-w8a8-emulation-gfx90a.patch`（4 个文件）里。

因此仓库只保留：`*.patch` + `bin/vllm-emulation`，并排除 `vllm/**`。
需要 overlay 时执行：

```bash
scripts/build_fp8_emulation_overlay.sh   # 从 site-packages 复制 + 打补丁
```

## 6. Git LFS 里到底放了什么

只有**不可再生**的二进制产出进 LFS（合计约 0.8 GB，GitHub Free 额度 10 GiB 存储 / 10 GiB 带宽）：

| 文件 | 大小 | 为什么值得占额度 |
|---|---|---|
| `hyperloom/reports/models/qwen38-27b-w8a8-dense/a8w8-fix/module_gemm_a8w8.gfx90a-fixed.so` | 23 MB | 在 gfx90a 上手工重建的 AITER CK 模块，是 `M<=64` 边界修复的可运行产物 |
| `…/a8w8-fix/module_gemm_a8w8.gfx90a-orig.so` | 23 MB | 同一模块的修复前基线，供 diff 与数值对照 |
| `…/profiling/benchmark_vllm_20260915_051456/torch_trace/*.pt.trace.json.gz` | 642 MB + 145 MB | decode 归因（60% 时间在 INT8 GEMM）的唯一原始测量数据，只能在该机该模型上重跑 |

不进 LFS 的（体积无信息量或可再生）：`models/*.safetensors`、`envs/**` 与
`hyperloom/envs/**` 的全部 `.so`、`hyperloom/session/**` 里的 worktree 与编译缓存、
`hyperloom/.tmp/**` 的 AOT 缓存、以及上面第 5 节里重复的 vLLM `.so`。
session 中的 run 状态与结果是**文本**，直接进 git 对象库（压缩后仅几 MB），不需要 LFS。

克隆时若想跳过下载：`GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 <url>`。

## 7. git-lfs 本机安装位置

本环境无法 `sudo`（no new privileges），git-lfs 静态二进制装在 **`./.tools/bin/git-lfs`**
（已 gitignore）。使用方式：

```bash
export PATH="$PWD/.tools/bin:$PATH"   # 之后 git add / commit / push 才会触发 LFS 过滤器
```

`scripts/bootstrap.sh lfs` 会在缺失时自动下载安装（git-lfs v3.8.0），并对当前克隆执行
`git lfs install --local`。**这一步不能省**：`.tools/` 不入库，全新克隆既没有 git-lfs
也没有 `filter.lfs.*` 配置，直接 checkout 只会得到 133 字节的指针文件。

## 8. hyperloom/session/：入库部分与遗留部分

session 是 optimizer 的运行现场，29 GB 里 28.9 GB 是"重放环境"
（每个候选一棵 vLLM worktree + `site-packages` + `boot/vcache` + `aot` +
`torch_compile_cache` + `triton_cache`），已用 `.gitignore` 的 session 专属规则剔除。
其余**状态与结果**（923 文件 / 49.9 MB）已入库，用于让 `hyperloom/reports/`
里每个数字都能落到原始记录上：critic/robustness 的 `request|emit|review|judge_bundle`
往返、每候选的 `config.yaml` 与 `benchmark_report.json`、`samples_gsm8k_*.jsonl`
逐样本精度明细、`baseline_config.with_envs.yaml` 实际生效的环境变量、
`patch_backups/*.bak` 改动前原文件、`reports/` 与 `agents/*/system_prompt*.md`。

sqlite `storage/coordinator.db` 与其 `-wal`/`-shm` **一并排除**：只提交 `.db` 会丢掉尚未
checkpoint 进主库的 wal 内容，得到一个半一致快照；同一批事件的文本形态已在
`*/emit.json`、`state.json` 里。需要时在本机只读导出：

```bash
python3 -c "import sqlite3;print('\n'.join(sqlite3.connect(
  'file:hyperloom/session/<模型>/<run>/storage/coordinator.db?mode=ro&uri=True').iterdump()))"
```

**权限遗留已清掉**：`hyperloom/session/` 下原有 305 个 `root:root 0600` 文件与 8 个 `0700`
的 `runtime/` 目录（hyperloom 在容器里以 root 身份写入本工作区），git 读不到就无法取哈希。
`scripts/fix_session_perms.sh --apply`（需 root，本 harness 的 sudo 被 no new privileges 挡住）
放开读位后，其中 266 个"状态与结果"已补入库：8 个 `manifest.json`、16 个 `state.json`、
5 个 `session_breakdown.json`、`reports/final.json`(5)、`optimization_journal.json`(7)、
30 个 specialist 的 `prompt.md`/`system_prompt.md`/`process.log`/`specialist_done.json` 等。
其余 2468 个文件全部落在 `runtime/`，见下一节——它们被**有意排除**，不是漏网。

## 9. 安全边界：什么东西绝对不能进这个仓库

本仓库是 **public**。下面是已经踩到过的线，写下来以免下一次 `git add hyperloom/session`
顺手把它们推上去：

| 位置 | 问题 | 处置 |
|---|---|---|
| `hyperloom/session/runtime/kernel-agent.env.sh` | `0644` 全局可读的运维 env 脚本，内含 `export ANTHROPIC_API_KEY=<48 字符真实密钥>` 与 `ANTHROPIC_BASE_URL` | `.gitignore` 排除整个 `hyperloom/session/**/runtime/`（提交 `236ba77`）；密钥值本身已核实**不在**任何已跟踪文件与任何历史提交中 |
| `.env`（工作区根） | 本机镜像 tag 与后续可能加入的凭据 | 只提交 `.env.example`；`*.env` 一律不入库 |
| `hyperloom/session/**/runtime/` 的其余内容 | `optimizer.lock`、`.audit.jsonl`、事件总线 spool、KB 预热缓存——工具活动状态，且 2419/2468 是 root `0600`，**无法逐一验密** | 同上排除；同等信息的文本形态已在 `*/emit.json`、`request.json`、`state.json` 里 |

因此约定：**不要用 `git add hyperloom/session` 这种整目录写法**（虽然 `runtime/` 已被忽略，
但这是给未来留的护栏）。要补 session 内容时用
`git ls-files --others --exclude-standard -- hyperloom/session | xargs -d '\n' -r git add --`，
提交前至少扫一遍凭据形态（`sk-ant-`/`ghp_`/`AKIA`/`PRIVATE KEY`）与本机那把密钥的指纹。

另有一条与本节无关但容易踩空的 git 行为：遍历撞上不可进入的目录（此处是那 8 个 `0700`
的 `runtime/`）会**中断该次递归**，表现为"只收到 1 个文件"而没有任何报错——这也是本仓库
改用显式 pathspec 清单的原因。
