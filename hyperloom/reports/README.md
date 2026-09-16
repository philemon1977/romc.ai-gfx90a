# 产出索引（按模型分开）

本目录里所有结论都归属到**具体模型**。两个模型性质完全不同，任何跨模型的推论都必须显式标注，不能混着讲。

| 目录 | 模型 | 性质 | 本仓库里的结论 |
|---|---|---|---|
| `models/qwen38-27b-w8a8-dense/` | `Qwen3.8-27B-ABLITERATED-W8A8-gdnint8` | **dense**，INT8 W8A8（compressed-tensors），27.9 GB，混合 `linear_attn` + `mtp` 层 | ① `baseline-validated-20260914.md` baseline 打通：444.9 tok/s/GPU，gsm8k 0.9674 ② `probe-concurrency-saturation.md` conc 32 起饱和 ~470–490 tok/s ③ `profiling-decode-attribution.md` decode 60% 在 INT8 GEMM（410 GB/s，仅其地板的 57%）④ `linear-backend-sweep.md` **负结论：配置级杠杆已穷尽**，剩余 +35% 属 kernel 级 ⑤ `gemm-35pct-verdict.md` 追 +35% 的杠杆逐条实测否决 ⑥ **`a8w8-M64-boundary-fix.md` 已取得 +17.6% 端到端**（AITER `M<64` 边界 off-by-one，改 `M<=64` 并重建 CK 模块；output_throughput 439.48→516.61，TPOT −16.1%，数值在容差内）。
&nbsp;&nbsp;&nbsp;同文件 C 一节是**用 harness 自身路径对同一修复的再测量**（444.86→512.55，+15.2%）＋ harness 精度门（gsm8k 0.9674 未变）。
&nbsp;&nbsp;&nbsp;**+17.6% 与 +15.2% 是一项修复、两把尺子，不可相加，也不是两笔成果。**<br>
&nbsp;&nbsp;&nbsp;⑦ `a8w8-tuning-PAUSED.md` —— A（aiter a8w8 调优，13,760 项）**已按指示暂停**；442 MB 构建缓存保留在容器层，恢复/校验清单在该文件 |
| `models/ornith-35b-a3b-moe/` | `Ornith-1.5-35B-A3B`（`Qwen3_5MoeForConditionalGeneration`） | **MoE**，未量化 bf16，68 GB，A3B（激活 3B） | 失败根因＝vLLM MoE 后端选择，非 ISA 问题 |
| `shared/` | 与模型无关 | 编排层缺陷、预算算术、容器/机架观测方法 | #1504 / #1505 / #1506 的归因 |

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
| `session/<模型>/<run>/…` | **状态与结果已入库**：`critic-workdir/`、`robustness-workdir/`、`runs/{baseline,specialist,integrate_patch}/` 下每候选的 `config.yaml`、`benchmark_report.json`、`samples_gsm8k_*.jsonl`、`baseline_config.with_envs.yaml`，以及 `reports/`、`agents/*/system_prompt*.md`。**不入库**的是重放环境——每个候选一棵 vLLM worktree + `site-packages` + AOT/inductor/triton 编译缓存（29 GB 里的 28.9 GB）。遗留：`manifest.json`/`state.json`/`session_breakdown.json` 等 267 个 root `0600` 文件本机读不到，放开方式见 `scripts/fix_session_perms.sh` |
| `envs/vllm/…`、`/home/qiba/ai/envs/wu1w*/…` | **不入库**的 Python 环境（12 GB 级）。重建见 `scripts/bootstrap.sh envs` 与 `docker/Dockerfile.vllm-local`，版本锁定见 `DEPENDENCIES.md` |
| `hyperloom/envs/vllm-fa/`、`hyperloom/logs/`、`hyperloom/.tmp/` | 同上：工具现场，不入库 |
| `hyperloom/patches/*/vllm/**` | 不入库的已构建 overlay（其 `.so` 与上游 wheel 逐字节相同）。改动本体是 `hyperloom/patches/*/*.patch`，重建用 `scripts/build_fp8_emulation_overlay.sh` |
| `module_gemm_a8w8.gfx90a-*.so`、`torch_trace/*.pt.trace.json.gz` | **已入库**，走 Git LFS；轻量克隆后需 `git lfs pull` 取回 |
| `scripts-local/…`、`kernels/…`、`kb/…`、`presets/…`、`patches-local/…` | **已入库**，路径一致 |

`profiling/benchmark_vllm_20260915_051456/torch_trace/` 里的 trace 用
[Magpie TraceLens](https://github.com/AMD-AGI/Magpie) 后处理，即可复现
`profiling-decode-attribution.md` 的归因表；`profiling/gap*/gap_analysis/*.csv` 是它的展开结果。
