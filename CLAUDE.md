# CLAUDE.md

本文件给在本仓库里工作的 AI agent（Claude Code / DSH / Codex / Cursor 等）。

**仓库性质**：本机 `8× AMD Instinct MI250 (gfx90a / CDNA2, 64 GB/GCD)` 上的 ROCm.AI 工作区 ——
Docker 环境栈 + Hyperloom 推理优化现场 + **可追溯的实测报告**。
仓库是 **public**（`git@github.com:philemon1977/romc.ai-gfx90a.git`），这一点决定了下面第 7 节。

**语言约定**：推理过程、文档、报告、提交信息一律**中文**（技术名词与路径保留英文）。
简单的问题思考一遍即可，复杂的问题需要思考两遍，综合两次思考的结果再回答。
所有生成的脚本和命令在执行之前都再检查审视一遍有没有疏漏、错误、风险。
如果脚本和命令执行结果出现错误，对于预期之外的问题，要多问几次为什么，直到找到根因，并且要对该问题的分析过程进行检查，看看有没有疏漏。

## 1. 硬件与运行时铁律（先读，避免踩已踩过的坑）

- **8 × MI250，gfx90a / CDNA2，双 die**。没有原生 FP8/FP4；`gfx90a` 上 **AITER 的 MoE 路径不可用**。
- **GPU0 上常有他人进程**（实测约占 53 GiB）。任何测试/起服务都显式挑卡：
  `HIP_VISIBLE_DEVICES=1`。
- **绝不抢卡、绝不杀别人起的服务、等对方释放后再起自己的**（用户 2026-09-18 定的规矩）。
  机制化的做法不是自觉，而是脚本：
  - `quark-int8/_guard.sh` —— `source` 后得到「本会话专用端口 + 只杀自己 PID 文件里的 pid +
    起服前硬门（别人有 api_server / 显存不足就直接退出，不腾地方）」。
  - `quark-int8/wait_gpu.sh` —— 轮询到某 GCD 有 ≥25 GB 空闲 HBM 才继续。
  - 已经出过事故：两个会话共用 8117 端口，PID 文件互相覆盖，停服时杀掉了对方服务。
- **只在需要时才起服务**（用户 2026-09-21 定的规矩）：起服前先说明这次为什么需要；测量/验证一跑完
  立即停服，不留常驻服务占着 8 张卡。报告进度时若服务是停的，要写明"已停、卡空闲"。
- **从现在起不能抢卡**（用户 2026-09-21 05:2x 定的规矩）：**任何占用 GPU 的运行（起服务、微基准、profiler）
  都要先取得用户明确许可**，不得自行判断"卡空着就能用"。需要 GPU 的结论一律先提方案、等批准。
  纯 CPU 的工作（读日志/源码、算账、写补丁、跑不碰 GPU 的分析）不受此限——本仓大量结论都能这样拿到。
- **HIP_VISIBLE_DEVICES 会泄进 ROCR 掩码子环境**，导致 `import vllm` 时直接抛
  （见 `hyperloom/reports/` 的 #1505 归因与 `hyperloom/patches-local/` 的 reconcile 补丁）。
- 一个模型的结论**不能**外推到另一个模型：dense INT8 与 MoE bf16 的失败长得一样，根因完全不同。

## 2. 第一次上手（还原不入库的资产）

工作区约 91 GB，仓库只装文档/代码/脚本/补丁/知识库/实测文本；其余由脚本还原。

```bash
# 轻量克隆：跳过 797 MB LFS
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 git@github.com:philemon1977/romc.ai-gfx90a.git ROCm.AI
cd ROCm.AI

scripts/bootstrap.sh                 # 交互式：lfs skills hyperloom envs images models smoke
scripts/bootstrap.sh --all           # 同上，非交互
scripts/bootstrap.sh lfs envs        # 只做某几步（lfs|skills|hyperloom|envs|images|models|smoke）

export PATH="$PWD/.tools/bin:$PATH"  # 必须：git-lfs 装在 .tools/bin（本机无 sudo）
git lfs pull                         # 需要 .so / trace 时；--include= 可只取一部分
```

`cp .env.example .env` 若缺失会被 bootstrap 自动补（`.env` 是本机配置，不入库）。

## 3. 日常命令

```bash
# 环境容器（工作区挂在容器 /workspace）
docker-compose run --rm pytorch                  # 或 tensorflow / jax / vllm
docker-compose run --rm -e HIP_VISIBLE_DEVICES=1 pytorch

# 冒烟（GPU 识别 + matmul / vLLM 端到端生成）
./scripts/smoke.sh                               # 全部
./scripts/smoke.sh pytorch                       # 单个：pytorch|tensorflow|jax|vllm

# vLLM 起 OpenAI 兼容服务（models/ 有 tiny Qwen2.5-0.5B）
docker-compose up -d vllm
docker-compose run --rm -e HIP_VISIBLE_DEVICES=1 vllm \
  vllm serve /models --served-model-name tiny-qwen --max-model-len 4096

# gfx90a FP8→BF16 dequant emulation overlay（从已提交的 .patch 重建 1.4 GB overlay 树）
scripts/build_fp8_emulation_overlay.sh

# 配方知识库：再生成 / 校验（KB 是 Hyperloom 的输入）
python3 scripts/seed_recipe_kb.py
python3 scripts/verify_recipe_kb.py
export KNOWLEDGE_LOCAL_ROOT=/home/qiba/ROCm.AI/hyperloom/kb   # 跑 Hyperloom 前必须

# 本地 vLLM 镜像重建
docker build -f docker/Dockerfile.vllm-local -t rocm-ai/vllm:0.28.0-rocm7.2.4 .
```

**测试**：本仓库没有 pytest/CI。验证靠两类东西 —— `tests/test_*.py`（由 `scripts/smoke.sh`
在容器里跑）与各报告目录下的复现命令/脚本。改动后按第 8 节清单跑。

## 4. 仓库地图

| 路径 | 内容 |
|---|---|
| `docker-compose.yml`、`.env.example`、`docker/` | 四个环境服务与本地 vLLM 镜像构建 |
| `scripts/` | `bootstrap.sh`、`smoke.sh`、`build_fp8_emulation_overlay.sh`、`fetch_models.sh`、KB 灌数/校验、`note_*.py` 现场取证 |
| `tests/` | 四个环境的冒烟脚本（容器内执行） |
| `hyperloom/patches/` | **gfx90a FP8→BF16 dequant emulation 补丁**（`*.patch` + `bin/vllm-emulation`） |
| `hyperloom/patches-local/` | ROCm 设备掩码 reconcile 补丁 |
| `hyperloom/kernels/` | 自写 gfx90a flash-decode / split-KV kernel、补丁与基准 |
| `hyperloom/kb/` | MI250x 配方知识库（7 元组目录契约，见第 6 节） |
| `hyperloom/presets/`、`scripts-local/`、`session-priors/` | 本机 preset、调参与探针脚本 |
| `hyperloom/session/` | optimizer 运行现场：**run 状态与结果入库**，重放环境不入库 |
| `hyperloom/reports/` | **实测报告与产出索引（入口：`hyperloom/reports/README.md`）** |
| `local-skills/mi250x-recipe-ops/` | 本机自建 Skill 及其数据 |
| `quark-int8/` | Quark INT8/INT4 路线探索：`RESULT.md`、`ROUTING_REPORT.md`、`LONGCONTEXT_FINDINGS.md`、`INT4_CT_PLAN.md`、`DSV41_CT_INT4_CONVERSION.md` + 大量 A/B 与测量脚本 |
| `amd-skills/`、`hyperloom/hyperloom/`、`hyperloom/kernelforge/`、`hyperloom/bin/` | **第三方代码，不入库**，bootstrap 还原（锁定版本见 `DEPENDENCIES.md`） |

`amd-skills/` 的技能已软链接到 `~/.dsh/skills/`、`~/.claude/skills/`、`~/.agents/skills/`。

## 5. 提交与写作规范

- 提交信息：`type(scope): 中文说明`，scope 用目录/主题（`quark-int8`、`kb`、`reports`、`kernels`、
  `session`、`repo`、`probes`、`guard`、`bootstrap`、`security`）。示例见 `git log`。
  一笔提交讲一件事；不确定的进展用 `wip(scope):`。
- 文档默认中文，**结论先行**，数字给出处（哪次 run、哪个文件、哪把尺子）。
- 新报告放 `hyperloom/reports/models/<模型>/`，并在 `hyperloom/reports/README.md` 的索引表里补一行
  （含结论一句话与文件链接）。跨模型推论必须显式标注。

## 6. 三处「契约」，改动前先确认

1. **补丁 vs overlay**：`hyperloom/patches/*/vllm/**` 的 1.4 GB overlay **不入库**，其 4 个 `.so`
   与上游 wheel **逐字节相同**——本项目对 vLLM 的改动**全在 `.patch` 里**。要改行为就改 patch，
   再用 `scripts/build_fp8_emulation_overlay.sh` 重建 overlay，**不要**直接编辑 overlay 树。
2. **知识库布局**：`hyperloom/kb/<model>/<hardware>/<framework_name>/<model_type>/<architectures>/<framework_version>/<precision>/recipe.json`
   是 Hyperloom local_store 契约，**勿手改目录结构**；生成入口是 `scripts/seed_recipe_kb.py`。
   同一 7 元组的多臂合并为一行，来源记在 `provenance.merged_from`。
   `hyperloom/session/knowledge/` 是旧容器会话（root 属主）写的旧根，**与本库无关，勿混用**。
3. **session 入库范围**：入库 = run 状态与结果（`manifest.json`、`state.json`、
   `critic-workdir/`、`robustness-workdir/`、每候选的 `config.yaml`/`benchmark_report.json`/
   `samples_gsm8k_*.jsonl`、`baseline_config.with_envs.yaml`、`patch_backups/*.bak`、agent 提示词快照）。
   不入库 = 重放环境（每候选一棵 vLLM worktree + 编译缓存，29 GB 里的 28.9 GB）与 `runtime/`。
   sqlite `coordinator.db` 与其 `-wal/-shm` **一起**排除（只提交 `.db` 会得到半一致快照）。

## 7. 安全红线（public 仓库）

- **不要写 `git add hyperloom/session`**。用显式清单：
  ```bash
  git ls-files --others --exclude-standard -- hyperloom/session | xargs -d '\n' -r git add --
  ```
- `hyperloom/session/**/runtime/` **一律不入库**，且这不是整洁性规则：
  `runtime/kernel-agent.env.sh` 是 0644 全局可读的脚本，内含 `ANTHROPIC_API_KEY` 与
  `ANTHROPIC_BASE_URL` 明文。该密钥值已核实不在任何已跟踪文件与历史提交中——保持这样。
- `.env`、`*.env` 不入库，只提交 `.env.example`。凭据形态提交前扫一遍：
  `sk-ant-` / `ghp_` / `AKIA` / `PRIVATE KEY`。
- 本机**没有 sudo**（no new privileges）：`scripts/fix_session_perms.sh --apply` 需要 root 才能
  放开 `session/` 下 root `0600` 文件的读位，当前 harness 做不了——遇到"读不到/取不到哈希"的
  session 文件，记下来而不是绕过去。
- LFS 只收**不可再生**的实测证据（AITER 修复产物 `.so`、profiler trace），额度有限（10 GiB），
  别把可再生的构建产物塞进去。

## 8. 改完什么，跑什么

| 改动 | 验证 |
|---|---|
| `docker-compose.yml` / `.env` / `docker/` | `./scripts/smoke.sh <env>`（注意先挑空闲卡） |
| `hyperloom/patches/**/*.patch` | `scripts/build_fp8_emulation_overlay.sh` → 用 `bin/vllm-emulation` 起服复测 |
| `hyperloom/kb/**` 或 `scripts/seed_recipe_kb.py` | `python3 scripts/seed_recipe_kb.py && python3 scripts/verify_recipe_kb.py` |
| `hyperloom/kernels/**` | 该目录下的 benchmark 脚本，与 baseline 同机同配置对比 |
| `quark-int8/**` 脚本 | 该目录对应的 `*_ab.sh` / 测量脚本；结论写回 `RESULT.md` 并标注日期与尺子 |
| 报告文档 | 与 `hyperloom/session/` 里的原始记录逐项对齐（数字必须能落到文件） |
| 新增/改动 `*.so`、`*.trace.json.gz` | `git lfs status`，确认走 LFS 而非普通提交 |

## 9. 证据与结论纪律（本仓库最容易被破坏的东西）

- 每个结论**归属到具体模型**：`hyperloom/reports/models/<模型>/`。跨模型推论要显式标注。
- **一项修复、两把尺子 ≠ 两笔成果**：同一修复用不同 harness 测出的 +17.6% 与 +15.2%
  **不可相加**。写报告时把「同一件事的两次测量」合并叙述。
- **负结论与撤回同样是产出**，要写清楚（例：`linear-backend-sweep.md` 的"配置级杠杆已穷尽"、
  被推翻的"MoE 缺 bf16 原子导致死"）。推翻既有结论时，在被推翻处留痕，不要静默改写。
- 结论里给到的路径若只在本机存在，在 `hyperloom/reports/README.md` 的路径对照表里补一行，
  写明仓库内对应物。
- 长跑实验用后台任务，日志与 PID 落到 `logs/`（含 `.pid`）或工具自己的目录；不要用
  "同分钟同名日志"这种会互相覆盖的写法——这正是第 1 节那起事故的成因之一。

## 10. 其它易踩的坑

- shell 里 `source .env` 是各脚本的既有约定（`scripts/smoke.sh` 如此），改脚本时保持一致。
- git 遍历撞上不可进入的目录（`session/**/runtime/` 的 0700）会**静默中断该次递归**，
  表现为"只收到 1 个文件"而无任何报错 → 这就是本仓库用显式 pathspec 清单的原因。
- 网络：`hub.docker.com` / `hf-mirror.com` 直连被 SSL 截断，用
  `docker pull dockerproxy.net/rocm/<img>:<tag>` 再 `docker tag`；模型走 ModelScope；
  PyPI 用清华源。见 README「下载加速经验」。
- 能用本地已有资产（conda env、已拉镜像、`.patch`）就不要拉新镜像/新权重。
