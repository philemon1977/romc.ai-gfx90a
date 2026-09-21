# 产出索引（按模型分开）

本目录里所有结论都归属到**具体模型**。两个模型性质完全不同，任何跨模型的推论都必须显式标注，不能混着讲。

| 目录 | 模型 | 性质 | 本仓库里的结论 |
|---|---|---|---|
| `models/qwen38-27b-w8a8-dense/` | `Qwen3.8-27B-ABLITERATED-W8A8-gdnint8` | **dense**，INT8 W8A8（compressed-tensors），27.9 GB，混合 `linear_attn` + `mtp` 层 | ① `baseline-validated-20260914.md` baseline 打通：444.9 tok/s/GPU，gsm8k 0.9674 ② `probe-concurrency-saturation.md` conc 32 起饱和 ~470–490 tok/s ③ `profiling-decode-attribution.md` decode 60% 在 INT8 GEMM（410 GB/s，仅其地板的 57%）④ `linear-backend-sweep.md` **负结论：配置级杠杆已穷尽**，剩余 +35% 属 kernel 级 ⑤ `gemm-35pct-verdict.md` 追 +35% 的杠杆逐条实测否决 ⑥ **`a8w8-M64-boundary-fix.md` 已取得 +17.6% 端到端**（AITER `M<64` 边界 off-by-one，改 `M<=64` 并重建 CK 模块；output_throughput 439.48→516.61，TPOT −16.1%，数值在容差内）。
&nbsp;&nbsp;&nbsp;同文件 C 一节是**用 harness 自身路径对同一修复的再测量**（444.86→512.55，+15.2%）＋ harness 精度门（gsm8k 0.9674 未变）。
&nbsp;&nbsp;&nbsp;**+17.6% 与 +15.2% 是一项修复、两把尺子，不可相加，也不是两笔成果。**<br>
&nbsp;&nbsp;&nbsp;⑦ `a8w8-tuning-PAUSED.md` —— A（aiter a8w8 调优，13,760 项）**已按指示暂停**；442 MB 构建缓存保留在容器层，恢复/校验清单在该文件 |
| `models/glm53-int4/` | `GLM-5.3-CT-Int4-W4A16`（`GlmMoeDsaForCausalLM`，78 层 = 3 dense + 75 稀疏 MoE，256 专家 top-8，gs=32） | **MoE**，CT INT4 W4A16，402 GB（每 rank 52.9 GiB），**DCP=8 / TP8 / 32K** | ① `moe-gemv-scale-hoist.md` —— **MoE decode GEMV 的真因是 group scale 的逐元素 gather**（不是 ALU、不是归约）：把 scale 提到 k 循环外后 gemm1 **8.7×**／gemm2 **5.0×**（生产分片形状），端到端**单流 6.4–6.8 → 9.6–10.7 tok/s**、**并发 32 聚合 33.8 → 59.1 tok/s**，事实召回 6/6 ② 同文记录两条负结论：**DCP=0 不快**、**MTP 投机净亏**（接受长度 1.18–1.67 但每步多跑一个完整 MTP 层，KV 池腰斩）③ `decode-step-attribution.md` —— **worker 内注入 torch.profiler**（唯一可行路径）拿到 decode 一步的 GPU 归属：稀疏注意力 **36%**／NCCL **19.4%**／非专家 W4A16 GEMM **18.9%**／MoE 专家 GEMV 只剩 **1.8%**；含一次「混窗口误判 prefill」的更正留痕；④ 定位教训：`rocprofv3` 与本配置在 RCCL 初始化处死锁；`torch.profiler` 在 driver 进程看不到 worker 的 kernel |
| `models/ornith-35b-a3b-moe/` | `Ornith-1.5-35B-A3B`（`Qwen3_5MoeForConditionalGeneration`） | **MoE**，未量化 bf16，68 GB，A3B（激活 3B） | 失败根因＝vLLM MoE 后端选择，非 ISA 问题 |
| `shared/` | 与模型无关 | 编排层缺陷、预算算术、容器/机架观测方法 | #1504 / #1505 / #1506 的归因 |
| `aiter-jit-prebuilt-reuse.md`（本目录顶层） | 与模型无关，**env 级基础设施**：aiter JIT 不做跨 env 复用，新 env 首次 import `module_gemm_a8w8` 现编 72 个 CK 实例 **≈50 min**；原生复用开关是 `AITER_JIT_DIR` + **裸名 `<md>.so`**。本机已建离线缓存 `~/.cache/aiter-gfx90a/`（426 MB），在 `vllm_master_rocm724` 实装后 import **0.300 s**、int8 GEMM 对拍 rel_err 6.4e-03/7.0e-03 通过。复用三判据＝aiter+torch+ROCm 版本一致且运行 arch 在 `.so` 的 `amdhsa--gfx*` 标记里；两个解读陷阱＝`max_abs` 0.5 是 bf16 一个 ULP（判据是相对误差）、缺调优表时走默认配置（不影响正确性，但会污染性能测量） |

| `mi250x-skill-scope-audit-2026-09-21.md`（本目录顶层） | 与模型无关，**技能层元工作**：审计 `/home/qiba/ai`（154 md + 53 配方 + 82 tools + 37 config），把 `mi250x-recipe-ops` 从「按会话日期堆叠的单文件」改成**九轴适用范围分层**（host/driver/rocm/torch/engine/arch/model/quant/topology × T0–T3） | ① **5 条配方从未进入 `data/*.json`**（`8121`、nightly-0918 env、Quark INT8 的 knob+op）——抽取链停在 09-15/16 ② 闸口三缺陷：`--dir` 不含 serving ⇒ **臂零审计**；`arms={a.get("id")}` 与真实键 `recipe_id` 不符 ⇒ **consumers 检查是死代码**；serving 路径 `frontmatter()` 直接 IndexError ③ ★ **`consumed_by` 不能用来反查适用性**：`8121` 被 0 个 knob 认领而实际适用 6 个；65 条 knob→臂边 **97% 单向** ⇒ 分家成 `applies_to`（谓词）／`consumed_by`（溯源）／`effect_by_scope`（分档效果），并落成可运行的 `scripts/scope_match.py` ④ **留痕更正两处既有判词**：MXFP4「机制不可达」错（有 `Mxfp4MoeBackend.EMULATION` 且本机实产过）；「int4 CK 重编到 gfx90a 不存在」错（源文写「不判死、可行」，实测编出码对象 58.3→127.9 TFlops），而 W4A8 反被否证（单流 M=1 时 int8 ALU 仅 **0.3%**）⑤ 拆分**零丢失**经两道验证（495/495 行逐字 + 33 个关键标识符存活）⑥ 诚实清单：`applies_to` 尚未回写权威 markdown（本轮先把它变成**可见的漂移**）、`env→版本` 三跳 join 第一跳就断（0/15 可解析） |

## 为什么必须分开

这两个模型的失败**看起来像同一个**（都是 `RuntimeError: Engine core initialization failed`、都是 `baseline 0.0`、都收在 `enablement_stalled`），但根因不同：

- **dense INT8** 死在**掩码泄漏**（`#1505`）：父进程 `HIP_VISIBLE_DEVICES=0..7` 泄进 ROCR 掩码的子环境，torch 在 `import vllm` 时直接抛，server 绑定端口之前就死。修法是 Magpie reconcile 补丁 / 源头 unset。
- **MoE bf16** 死在 **vLLM 的 MoE 后端选择**：`kernel_config` 里 `moe_backend='auto'` → 优先级表把 AITER 排第一 → `VLLM_ROCM_USE_AITER_MOE` 默认 `True` 导致强行选中 AITER → 而 gfx90a 是 CDNA2、AITER 不可用 → `ValueError` 硬失败且**不回退 Triton**。与掩码、与原子指令都无关。

把这两条混在一起讲，就会得出"gfx90a 上 MoE 路径死（缺 bf16 原子）"这种错误结论——那正是 `shared/` 里记录的一处**已被推翻**的判断。

## 历史文件

`github-issue-{1,2,3}-*.md`、`mission-final-20260914.md` 是早期产出，写作时未按模型分账：#1504 的现场是 dense INT8 session，但同一类失败也命中过 Ornith MoE session；#1505 的现场是 dense。阅读时以本索引的归属为准。

## 读这些报告时的路径说明

报告是当时在那台 8×MI250 上写的，正文里夹着若干**只在该机器上存在**的路径。它们不是产出的替代品——结论、数字、配置、补丁都在本目录里，路径只是取证出处：

| 报告里出现的路径 | 在仓库里的对应物 |
|---|---|
| `session/<模型>/<run>/…` | **状态与结果全部已入库**：每个 run 的 `manifest.json`、`state.json`、`session_breakdown.json`、`reports/{final,optimization_journal}.json`，`critic-workdir/`、`robustness-workdir/`，`runs/{baseline,specialist,integrate_patch}/` 下每候选的 `config.yaml`、`benchmark_report.json`、`samples_gsm8k_*.jsonl`、`baseline_config.with_envs.yaml`、`specialist_done.json`、`prompt.md`/`system_prompt.md`/`process.log`。**不入库**两类：① 重放环境——每候选一棵 vLLM worktree + `site-packages` + AOT/inductor/triton 编译缓存（29 GB 里的 28.9 GB）；② `runtime/`——工具活动状态，且其中含明文密钥，见 `DEPENDENCIES.md` 第 9 节 |
| `envs/vllm/…`、`/home/qiba/ai/envs/wu1w*/…` | **不入库**的 Python 环境（12 GB 级）。重建见 `scripts/bootstrap.sh envs` 与 `docker/Dockerfile.vllm-local`，版本锁定见 `DEPENDENCIES.md` |
| `hyperloom/envs/vllm-fa/`、`hyperloom/logs/`、`hyperloom/.tmp/` | 同上：工具现场，不入库 |
| `hyperloom/patches/*/vllm/**` | 不入库的已构建 overlay（其 `.so` 与上游 wheel 逐字节相同）。改动本体是 `hyperloom/patches/*/*.patch`，重建用 `scripts/build_fp8_emulation_overlay.sh` |
| `module_gemm_a8w8.gfx90a-*.so`、`torch_trace/*.pt.trace.json.gz` | **已入库**，走 Git LFS；轻量克隆后需 `git lfs pull` 取回 |
| `scripts-local/…`、`kernels/…`、`kb/…`、`presets/…`、`patches-local/…` | **已入库**，路径一致 |
| `aiter/jit/module_gemm_a8w8.so`、`aiter/jit/build/**/*.o` | **不入库**的 JIT 编译产物（本机 gfx90a 全量集 187 MB，72 个 CK 实例；现场重编一次约 50 min）。离线缓存放在仓库外的 `~/.cache/aiter-gfx90a/`，`ARCHIVE.md` 记了复用机制与前置判据，`archive.py`/`restore.py` 负责归档与恢复。**判据**：aiter 源树 + torch + ROCm 三者的版本都要与归档时一致，且运行 arch 在 `.so` 的 `amdhsa--gfx*` 标记里；装完必须过一次 import + 单形状 int8 GEMM 对拍才算数 |

`profiling/benchmark_vllm_20260915_051456/torch_trace/` 里的 trace 用
[Magpie TraceLens](https://github.com/AMD-AGI/Magpie) 后处理，即可复现
`profiling-decode-attribution.md` 的归因表；`profiling/gap*/gap_analysis/*.csv` 是它的展开结果。

## 2026-09-21 追加：GLM-5.3 int4 decode 配置消融（新报告）

| 结论一句话 | 文件 |
|---|---|
| `DSV41_IDX_AITER_KERNEL=1` 是当天唯一显著正向（conc32 +69.3%，且默认值是更慢更不可信那一支）；split-K 三档全负；QuickReduce 固定吃 ~9 GiB/卡不可用；DCP=8 不省显存但把每 token KV 单价降到 1/8（1M 需再叠 fp8 KV） | [`models/glm53-int4/decode-config-ablation.md`](models/glm53-int4/decode-config-ablation.md) |
| QR C2+C3 三臂实证（A stock / A2 含补丁但 env 不设 / B QR 真开）：**QR 开在本模型起不来** —— `init_custom_qr` 在显存规划前吃掉 ~9 GiB/卡，0.97 档下 `request_memory` 抛 `Free memory 54.9 < 62.06 GiB`；要开必须 util ≤0.858、KV 只剩 ~2 GiB（1M 不可能）⇒ 结论与上一行一致，此处给出机制与数字 | [`models/glm53-int4/qr-c2c3-verdict.md`](models/glm53-int4/qr-c2c3-verdict.md) |

**本机路径 ↔ 仓库内对应物**（§9 约定）：

| 本机路径 | 仓库内对应物 |
|---|---|
| `/home/qiba/ai/models/ZhipuAI/launcher/glm53_int4w4a16_vllm_rocmnightly0918_32k_8121_mi250dx8.sh` | `quark-int8/launchers/`（同名镜像；今天改了 QR 条件注入、cudagraph capture 防御两处） |
| `rocm-ai/vllm:glm53-int4-gfx90a-0918-qr`（docker 镜像，含 QR C2+C3） | applier 在 `hyperloom/patches-local/apply_gfx90a_quickreduce.py`；镜像本身需 `docker commit --change` 重建，见报告第五节 |
| `quark-int8/logs/stack/results.jsonl`（gitignore） | `hyperloom/reports/models/glm53-int4/stack_results_20260921.jsonl` |
| `/home/qiba/ai/docs/recipes/**/*.md`（53 条配方，**不在 git**） | `mi250x-recipe-ops` 技能的 `data/*.json` 是它的合成物；权威仍是本机 markdown |
| `/home/qiba/ai/tools/audit_{recipes,ports,conf_wiring,log_paths,ctx,kill_discipline,studio_llama_pin}.py`、`instruction_budget_check.sh`、`verify_env_patches.sh` | 无仓库内对应物（`audit_recipes.py` 的**技能侧**对应物是 `scripts/audit_skill_recipes.py`） |
| `/home/qiba/ai/config/{ports.conf,env-lock/*.txt,gpu_mi250x_*.conf,moe-tuned/}` | 无仓库内对应物；`ports.conf` 是端口权威，`env-lock` 是版本证据（**不含 ROCm 版本**） |

