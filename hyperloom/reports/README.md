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
