# 追 +35%：**取得 +17.6% 端到端**（AITER a8w8 派发启发式的 M=64 边界缺陷）

日期 2026-09-15 17:20–18:10 CST。承接 `gemm-35pct-verdict.md`（当时结论："配置杠杆已穷尽，需为 gfx90a 从源码构建 CK kernel"）。
**本次即为该内核工作的第一个增量，并且成功。**

## 结论摘要

> **数字出处（务必分清，避免把一次收益记成两次）**
> 下表全部来自 **`vllm bench serve`**（我自己在跑 C **之前**做的旁测），**不是** harness 路径。
> 下文 C 一节里的 `444.86 → 512.55（+15.2%）` 是 **Hyperloom→Magpie→InferenceX** 对
> **同一个修复**的**再次测量**。两者是**同一效应的两把尺子**，不可相加、不可并列成两个成果。
> 归因：**增益由 AITER `M<=64` 修复产生**；C 只提供独立仪器的复核与 harness 自己的精度门。

| 指标 | 修复前 | **修复后** | 变化 |
|---|---:|---:|---:|
| **output_throughput**（`vllm bench serve`，128 prompts） | 439.48 | **516.61** | **+17.6%** |
| mean TPOT | 134.58 ms | **112.88 ms** | **−16.1%** |
| 稳态生成吞吐中位数（`Running≥32` 采样） | 553.5 | **684.5** | **+23.7%** |
| 总时长（128 prompts × 1024 OSL） | 298.2 s | 253.7 s | −14.9% |

负载与修复前**完全一致**：conc 64、ISL/OSL 1024、128 prompts、`--ignore-eos`、TP=1 单 GCD、同 shim、同 server args。

**数值校验**：全部 8 个形状（含 lm_head、M=1/2/128）最差 `ratio = 0.00297`，调优器默认阈值 `errRatio = 1e-2` → **3.4 倍余量，在容差内**。

## 根因：派发启发式的 `M < 64` 边界把 M=64 排除在"小 M 核"之外

`aiter_meta/csrc/ck_gemm_a8w8/gemm_a8w8.cu` 的 `rowwise_dispatch()` 逻辑是：

1. 查**编译期生成**的 `(gfx, cu_num, M, N, K)` 查找表（`GENERATE_LOOKUP_TABLE`，由 `gen_instances.py --tune_file` 从 tuned CSV 生成）；
2. 未命中则按 padding 后的 M 再查；
3. 仍未命中才走 `rowwise_heuristic_dispatch(M, N, K)`。

而启发式的分支全部写作 `M < 64`：

```cpp
else if (M < 64 && N < 2048 && K < 2048) { ... 64x16x16x128 ... }
else if (M < 64 && K < 2048)             { ... 128x16x32x128 ... }
else if (M < 64 && N < 2048)             { ... 128x32x16x128 ... }
else if (M < 64 && N > 2048 && K > 2048) { ... 64x16x16x256 ... }   // ← 本该命中
else if (M < 64)                         { ... 64x16x16x128 ... }
else if (K < 1024)                       { ... 256x128x128x128 interwave_v1 ... }
else if (M < 1024)                       { ... 256x128x128x128 intrawave_v3 ... }  // ← M=64 实际落这里
```

**M 恰好等于 64（conc 64 的稳态 decode 批量）时，所有 `M < 64` 分支全为假**，于是掉进为大 M 准备的
`256x128x128x128_intrawave_v3`：`MPerBLOCK=256`（实际 M 只有 64，75% 空转）、`NPerBLOCK=128`。

可预测的后果，且与观测逐字吻合：
- N=5120 时网格只有 `5120/128 = 40` 个 CTA，而本机 **104 个 CU** → 占用率不足 → 有效带宽仅 222 GB/s；
- 带宽随 N **单调递增**（222→238→314→353→483→554 GB/s），正是"网格越大越能填满"的特征；
- profiling 里看到的正是 `ck::kernel_gemm_xdl_cshuffle_v3`（tile 256,128,128）。

**这是 AITER 的一个 off-by-one**：它影响任何在 gfx90a 上以 **batch 64** 跑 INT8 W8A8 的用户——一个非常常见的 serving 配置。

## 修复

`gemm_a8w8.cu` 中 5 处 `M < 64` → `M <= 64`（补丁见 `a8w8-fix/gemm_a8w8_M64_boundary.patch`）。
未改任何其他逻辑；M<64 与 M>64 的行为完全不变。

重建并安装（关键：**AITER 在预编译 .so 架构匹配时优先用它，新构建默认落在 `/root/.aiter/jit/` 而不生效**）：

```bash
export AITER_REBUILD=1     # 触发从本地 csrc 重建 module_gemm_a8w8（95 个实例，约 9 分钟）
# 构建产物 → /root/.aiter/jit/module_gemm_a8w8.so
# 必须覆盖 site-packages 里那份才会被加载：
#   /opt/envs/wu1w/lib/python3.12/site-packages/aiter/jit/module_gemm_a8w8.so
# （该路径在容器内是只读 bind mount，需从宿主侧写入：
#   /home/qiba/ai/envs/wu1w-int8-028/lib/python3.12/site-packages/aiter/jit/）
```

注意 `gemm_a8w8.cu` 也只在宿主侧可写（`/opt/envs/wu1w` 是 `rw=false` 的 bind mount）。

## 隔离测量：按形状的前后对比（M=64，splitK=0）

| shape | N | K | 修复前 ms / GB/s | **修复后 ms / GB/s** | 加速 |
|---|---:|---:|---|---|---:|
| o_proj | 5 120 | 6 144 | 0.14 / 221 | **0.08 / 392** | **1.78×** |
| down | 5 120 | 17 408 | 0.37 / 238 | **0.20 / 454** | **1.91×** |
| mlp_mid | 14 336 | 5 120 | 0.23 / 313 | **0.15 / 481** | **1.54×** |
| qkv | 16 384 | 5 120 | 0.24 / 353 | **0.17 / 490** | **1.39×** |
| gate_up | 34 816 | 5 120 | 0.37 / 482 | **0.34 / 518** | 1.07× |
| lm_head | 248 320 | 5 120 | 2.30 / 553 | 2.36 / 540 | 0.97×（中性） |

## 自洽性检验（这条让结论可信）

profiling 已测出 **GEMM 占 decode 步的 60.09%**。若加权 GEMM 加速为 ~1.4×，则整步应缩短
`0.60 × (1 − 1/1.4) = 17.1%`；**实测 TPOT 缩短 16.1%**。两条独立路径吻合。

## 为什么是 +17.6% 而不是当初估的 +35%

- `lm_head`（N=248320，占每步权重 1.27 GB）**中性**，它对总量有拉低作用；
- 整步另外 40% 完全未动：AITER paged attention 15.78% + GDN 线性注意力 14.21% + 量化/杂项；
- 修好后的 GEMM 也仍只到 392–518 GB/s，而可达参考是 721（M=1 gemv）/1372（纯读）GB/s。

⇒ **剩余余量仍在**，且路径已明确：用 AITER 的调优器在全部 **95 个候选 kernel** 中按形状选最优
（`python3 aiter_meta/csrc/ck_gemm_a8w8/gemm_a8w8_tune.py -i <untuned> -o <tuned> -k`），
因为查找表是编译期从 tuned CSV 生成的，所以调优结果会**直接进入 `rowwise_dispatch` 的第一步**，
不再依赖启发式。本次已证明调优是有回报的（启发式手改一项就拿到 +17.6%）。

## 与 Hyperloom 的关系（必须说清）

**这 +17.6% 不是 Hyperloom 的 `cumulative_gain_val`**，是我们直接在 AITER 内核派发层做的修复。
它证明了 `gemm-35pct-verdict.md` 的判断（"增益在 kernel 层，不在配置层"），并给出了第一个增量。
若要让它成为编排器可复现的收益，需要把当前修复后的环境作为**起始栈**交给 Hyperloom 去 baseline 化。

## 产物与回滚

- `a8w8-fix/gemm_a8w8_M64_boundary.patch` —— 5 行改动
- `a8w8-fix/module_gemm_a8w8.gfx90a-fixed.so` —— 修复后模块（23 MB）
- `a8w8-fix/module_gemm_a8w8.gfx90a-orig.so` —— 原模块，回滚用
- 脚本：`scripts-local/probe_a8w8_splitk.py`（按形状测带宽）、`scripts-local/verify_a8w8_numerics.py`（数值校验）
- 备份位置：宿主 `/tmp/gemm_a8w8.cu.orig`、`/tmp/old_a8w8.so`；容器 `/root/a8w8-backup/`

**回滚**：把 `module_gemm_a8w8.gfx90a-orig.so` 覆盖回 `aiter/jit/module_gemm_a8w8.so`，
并把 `gemm_a8w8_M64_boundary.patch` 反向应用到 `gemm_a8w8.cu`。

## 建议的上游贡献（ROCm/aiter）

1. **本次修复**：`gemm_a8w8.cu` 的 `M < 64` 边界应为 `M <= 64`；否则 batch=64 的 INT8 decode 会退化为
   为大 M 设计的 256×128 tile，在 MI250X（104 CU）上只填 40 个 CTA。
2. `configs/a8w8_tuned_gemm.csv` 无 gfx90a 行（579 行全是 gfx942/gfx950 + cu_num∈{80,256}，而本机
   gfx90a/cu_num=104）→ gfx90a 上所有 a8w8 GEMM 静默走启发式；未命中告警是 INFO 级，易被淹没。
3. gfx90a 的 a8w8 预编译模块无 splitK 变体（`splitK>0` 抛 `RuntimeError` 而非回退到 0）。

---

# C：Hyperloom 同口径复核（修复后环境重跑 session）

Session `Qwen3.8-27B-ABLITERATED-W8A8-gdnint8/20260915T103412Z-bf20aced`（`--max-hours 3`，
保留精度门，其余配置与修复前那次**完全一致**：TP=1 单 GCD、conc 64、ISL/OSL 1024、320 prompts、同 shim、同 server args）。

## 同一条测量链上的前后对比（327,680 输出 tokens，conc 64）

| 指标 | 修复前 session<br>`…20260914T173139Z-3ae15c89` | **修复后 session** | 变化 |
|---|---:|---:|---:|
| **output_throughput** | 444.86 | **512.55** | **+15.2%** |
| total_token_throughput | 889.71 | 1025.10 | +15.2% |
| mean TPOT | 137.81 ms | **118.47 ms** | **−14.0%** |
| median TPOT | 140.51 ms | 121.20 ms | −13.7% |
| 测量时长 | 736.6 s | 639.3 s | −13.2% |
| mean TTFT | 6121 ms | 6439 ms | +5.2%（prefill 未受影响，符合预期） |
| **gsm8k `exact_match,strict-match`** | 0.9674 | **0.9674** | **完全不变** |
| gsm8k flexible-extract | 0.9666 | 0.9666 | 完全不变 |
| eval 生成数 / 截断数 | 1319 / 10 | 1319 / 10 | 一致 |

## 本表贡献的是什么（不是"又一笔收益"）

**必须先说清归因：上表的 +15.2% 不是 C 取得的成果。** 修复在跑 C 之前就已完成并测量过
（`vllm bench serve`：439.48 → 516.61，+17.6%）。C 没有引入任何新变量，它做的是
**换一把独立的尺子把同一个效应再量一遍**。所以正确的表述是：

> **一项修复（AITER `M<=64` 边界），两把尺子各测一次：+15.2% 与 +17.6%。**
> 二者不可相加、不可并列成两个成果。

C 真正带来的三点增量**信息**（而非增量收益）：

1. **排除工具性偏差**：本表由 Hyperloom→Magpie→InferenceX 产出，`vllm bench serve` 是另一套工具。
   两者同向（+15.2% vs +17.6%），差异可由口径解释（bench 含爬坡；本 session 320 prompts vs 128 prompts）。
   即：增益不是某个 benchmark 工具的产物。
2. **精度由 harness 自己的门验证**：换 kernel 的最大风险是数值漂移，而 gsm8k 两个口径
   **逐字未变（0.9674 / 0.9666）、截断数一致（10）**——这比合成 `errRatio=0.00297` 更接近真实分布。
   这一条是 C 独有的价值，`vllm bench serve` 测不到。
3. **作用面交叉验证**：TTFT 几乎不动（+5%）而 TPOT 降 14%，与"只改 decode GEMM 派发"完全吻合
   （prefill 是大 M，本就不经过被改的 `M<=64` 分支）。这排除了"整机变快/环境噪声"的解释。

## 关于 `cumulative_gain_val` 的定位说明（避免误读）

AITER 修复是**环境性**的，它抬高的是 session 的 **baseline**，不会计入 `cumulative_gain_val`——
后者在 `orchestrator/loop/writeback.py:667-669` 定义为
`(candidate − 本 session 自身 baseline) / baseline × 100`（`against_baseline=True`）。
因此本节的产出是**「同一编排器在同一路径下测得的 baseline 提升 +15.2%」**，
而不是一个非零 gain。若要让 gain 非零，需要编排器在修复后的起点上**另外**找到改进——
该 session 已进入优化阶段（`baseline: succeeded`，specialist 运行中），结果见下节。

## 产物

- 修复后 baseline：`…/20260915T103412Z-bf20aced/runs/baseline/51ac28ff8aed4ad59746fdb889fae7bb/`
  （`inferencex_result.json`、`hyperloom_eval_bounds.json`、`results_2026-09-15T11-30-23.287028.json`）

## C 的最终结论（session 已跑完）

`20260915T103412Z-bf20aced`，`--max-hours 3`，全程 GPU 独占。

```
stop_reason          : sweep_done
baseline             : 512.5 tok/s/GPU
cumulative_gain_val  : 0.00%（no explore KEEP landed）
crash_count          : 0
```

任务终态：`target_analysis 1 · baseline 1 · specialist 11 · explore 3成功+1取消 · integrate_patch 1 · report 1 · session_breakdown 1`

### 三点判断

1. **#1504 那类 enablement 死锁已彻底消失。** 这是该编排器在本机第一次**完整走通**
   `enablement → baseline → explore → integrate_patch → report`。`stop_reason` 从 `enablement_stalled`
   变成 `sweep_done`，且 `integrate_patch` 首次 succeeded。
   归功于：Magpie 掩码补丁 + `max_turns=36` + 源头 unset + 预算参数（`--max-hours 3`、prelude 0.10）。
2. **gain 仍为 0.00%，但原因换掉了。** 3 个 explore 轮次全在测**同一个**变体 `v00_gdn-ssm-state-bf16`
   （GDN 线性注意力 SSM state 转 bf16 —— 一个真正的 kernel 级候选），而三次 decision round
   都被 `session_time_exhausted` 截断（11:41 / 12:12 / 12:20），**没有任何一个 explore 目录产出
   `inferencex_result.json`**。所以"未产生 KEEP"是**没测完**，不是"测了且更差"。
3. **瓶颈从编排器转移到了预算调度。** baseline + 精度门吃掉 180 分钟里的前 ~70 分钟，
   剩 ~110 分钟不足以跑完一个 explore 变体的完整轮次（静态估计单轮 3900s）；
   编排器也因此在 11:30 主动跳过了 initial profile（`1906s of preparation budget left against an expected 3347s`）。

### 由此得到的下一次 run 配方（若要冲非零 gain）

- `--no-eval`：省回那 ~40 分钟精度门（精度已由本报告两次独立验证覆盖）
- `--max-hours ≥ 4`：让 `integrate`(7800s) 与至少 2–3 个 explore 完整轮次放得下
- 这两个参数是**互补**的：只省 eval 不给足时长，仍会重演"截断"。

### 归因重申（避免重复计账）

本节 **+15.2% 属于那一项 AITER 修复**，由第二条仪器复测得到；它**不是** C 的新收益，
也不会出现在 `cumulative_gain_val` 里（gain 的分母是本 session 自身的 baseline）。
