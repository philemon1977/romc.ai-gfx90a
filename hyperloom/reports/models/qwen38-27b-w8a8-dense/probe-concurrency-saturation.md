# 探针报告：`Qwen3.8-27B-ABLITERATED-W8A8-gdnint8`（dense INT8）的并发饱和点

日期 2026-09-15 04:30–05:05 UTC。目的：在决定是否重跑长 session 之前，先量出**这个模型在这台机器上的可达上限，以及瓶颈在哪**。

配置与 baseline 完全一致（同镜像、同 shim、同 server args、同 `ROCR_VISIBLE_DEVICES=0`、`--max-model-len 6144`、AITER 开启、`cudagraph_mode=FULL_DECODE_ONLY`）。仅并发可变。

模型结构（`config.json` → `text_config`）：**64 层 = 48 层 `linear_attention` + 16 层 `full_attention`**（混合注意力），24 attn heads / **4 KV heads** / **head_dim 256** / hidden 5120 → 每个 token 的 KV ≈ 2×4×256×2B ×16 层 = **64 KB**。

## 探针 A：实测可达 HBM 带宽（单 GCD）

脚本 `scripts-local/probe_hbm_bw.py`。

| 负载 | 实测 | 相对标称 1.6 TB/s |
|---|---|---|
| `sum`（纯读） | **1372 GB/s** | 86% |
| `copy`（读+写） | 1224 GB/s | — |
| **`gemv`（读权重为主，decode 形状）** | **721 GB/s** | 45% |

**要点：纯读能到 1372，但 decode 形状（GEMV）只到 721。** 所以解释本模型时正确的带宽分母是 **721–839 GB/s，不是 1372**。

## 探针 B：并发扫描（`vllm bench serve`，ISL/OSL = 1024/1024，`--ignore-eos`）

脚本 `scripts-local/probe_conc_sweep.sh`；原始结果 `probe/conc_sweep.csv`、`probe/conc*.json`。

| conc | prompts | **output tok/s** | total tok/s | TPOT mean | TTFT mean | duration |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 29.24 | 58.48 | 33.25 ms | 1 003 ms | 70.0 s |
| 8 | 16 | 196.78 | 393.55 | 38.22 ms | 2 492 ms | 83.3 s |
| 32 | 64 | 468.79 | 937.58 | 62.73 ms | 5 507 ms | 139.8 s |
| **64**（baseline 点） | 128 | **473.78** | 947.56 | 124.01 ms | 10 910 ms | 276.7 s |
| 128 | 128 | 491.66 | 983.32 | 181.46 ms | **39 358 ms** | 266.6 s |

### 三条结论

**1. 聚合吞吐从 conc 32 起就饱和：468.8 → 473.8 → 491.7 tok/s。**
并发翻 4 倍（32→128）只换来 **+4.9%**。而 TPOT 几乎严格随并发倍增（62.7 → 124.0 → 181.5 ms），即每流吞吐被精确抵消——这就是饱和的定义。

**2. baseline 的测量点（conc 64，444.9 tok/s）落在平台区中段，且可复现。**
本次同点测得 473.78（差异来自 prompt 数与 warmup）。**结论是指标饱和，不是那次测量错误。**

**3. 单流已经跑在可达带宽上；问题在于 batch 增大时每步成本不摊薄。**

| conc | 每步耗时 | 权重读 27.9 GB 的有效带宽 |
|---:|---:|---:|
| 1 | 33.25 ms | **839 GB/s**（≈探针 gemv 的 721，甚至更好） |
| 8 | 38.22 ms | 730 GB/s |
| 32 | 62.73 ms | 445 GB/s |
| 64 | 124.01 ms | 225 GB/s（含 KV 约 6.3 GB → 276 GB/s） |
| 128 | 181.46 ms | 154 GB/s（含 KV 约 12 GB → 222 GB/s） |

decode 每步都要把权重读一遍，**与 batch 无关**。所以 batch 增大本该摊薄权重读取、让聚合吞吐线性上升：

> 若 conc 64 的每步仍是单流的 33.25 ms，聚合应为 64 / 0.03325 ≈ **1 925 tok/s**；实测 474 tok/s → **约 4.1 倍被丢掉**。

### 已排除的解释

- **不是 KV 容量导致的抢占**：`grep -ci "preempt|swap|recompute"` = **0**。server 日志里没有一次抢占/换出/重算。（顺带记录：引擎报 `Available KV cache memory: 30.2 GiB`、`GPU KV cache size: 276,918 tokens`、`Maximum concurrency for 6,144 tokens per request: 45.07x`——baseline 的 conc 64 看上去越过了这个名义拐点，但既然零抢占，它就不是瓶颈。）
- **不是纯带宽**：conc 64 时把权重 27.9 GB + KV 6.3 GB 一起算，也只有 276 GB/s，是探针可达值（721–839）的 33–38%。
- **不是 TTFT/排队**：TTFT 确实从 1.0 s 爆到 39.4 s（conc 128），但 `output_throughput` 是稳态解码指标，已被 TPOT 独立印证。

### 尚未排除的假设（需要 profiler）

每步成本几乎**线性随 batch 增长**，指向"每个序列都要付且不随 batch 摊薄"的工作。本模型有两处高危嫌疑，都与既有发现吻合：

1. **16 层 `full_attention`，head_dim = 256。** 此前已确认：本模型 head_dim=256 **被 AITER FA decoder 路径拒绝**，AITER FA 只命中 ViT/MMEncoder → 即注意力落在非 ASM 的通用路径上。hd256 + GQA(4 KV heads) 在 gfx90a 的通用 kernel 上批量效率可能很差。
2. **48 层 `linear_attention`（GDN）** 的 decode 路径（在 MoE 模型上我们已看到它回退到 Triton）。

定位方法很直接：**在 conc 64 下抓一次 decode step 的 torch trace，用 TraceLens 归因到 kernel**。这正是 `magpie-kernel-evaluator` / `tracelens-analysis-orchestrator` 两个 skill 的用途。

## 对"要不要重跑长 session"的决策含义

**建议：不要按原计划重跑。** 依据是本探针把关键杠杆空间量完了：

1. **concurrency/config 空间已穷尽**：conc 32→128 全在 ~470–490 的平台区。这个空间里没有任何配置能买到 30%。
2. **剩余余量（约 4.1 倍）在"大 batch 下每步 kernel 效率"，而那恰好是 `--no-kernel` 关掉的杠杆。** 原先提议的 `--no-eval --max-hours 4 --no-kernel` 重跑，在结构上就没有通往 gain 的路径——上一次的 0.00% 会以**同一个**原因重演，只是这次原因是量化的、可预期的。
3. **要追这 4.1 倍，正确的下一步是一次约 40–60 分钟的 profiling**（conc 64 的 decode step → TraceLens → 点名 kernel），然后带着**具体的 kernel 级目标**、并且**打开 kernel 杠杆**去跑 Hyperloom。否则长 session 只是把 3 小时花在一个不存在的配置杠杆上。

## 产物

- `probe/conc_sweep.csv`、`probe/conc{1,8,32,64,128}.json`、`probe/conc*.log`
- `scripts-local/probe_hbm_bw.py`、`scripts-local/probe_conc_sweep.sh`
- server 侧日志：容器内 `/tmp/probe-b-server.log`

## 本报告修正了什么

- 早前用 **1372 GB/s** 当分母，推出"conc 64 只有 12% 带宽利用率、缺口 7 倍"。**分母错了**：decode 形状的可达带宽是 **721–839 GB/s**，据此单流已经贴着上限。缺口应从"相对 roofline 的 7 倍"改述为"**batch 未摊薄带来的约 4.1 倍**"。方向（存在大余量）不变，但数值和归因都改了，且余量**不在配置层**。
- 早前的猜测"conc 64 越过了 KV 容量拐点导致抢占"——**被证伪**（零抢占）。
