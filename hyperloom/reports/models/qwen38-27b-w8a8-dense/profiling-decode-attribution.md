# Profiling 归因：`Qwen3.8-27B-ABLITERATED-W8A8-gdnint8`（dense INT8）在 conc 64 的 decode 步

日期 2026-09-15 05:10–05:40 UTC。目的：把探针量到的"batch 未摊薄损失"归因到**具体 kernel**。

- 工具：Magpie `benchmark`（torch profiler）+ Magpie **standalone gap analysis**（`--trace-dir`）
- 干净的未 profiling 基线仍在：探针 B 的 conc 64 = **473.78 tok/s**（本报告只做归因，不用 profiling 的数字当性能结论——skill 明确警告 profiler 会扰动延迟）
- 配置：`scripts-local/magpie-profile-conc64.yaml`（TP=1、单 GCD、ISL/OSL 1024、conc 64、`NUM_PROMPTS 64`、`RUN_EVAL=false`）
- Trace：112 MB，`profiling/benchmark_vllm_20260915_051456/torch_trace/`

## 方法与一个必要的修正

第一次 gap analysis 用默认全窗口（0–100%），结论是"GEMM 占 66%、注意力只占 11%"。**这个窗口不能用**：CSV 的 `Input Shapes` 显示那批 GEMM 的 M 是 **542 / 784 / 1266 / 1812**，即 **chunked-prefill 的累积 token 数**，decode 形状（M=64）只出现过一次（`[64,5120]x[248320,5120]`，LM head）。全窗口把 prefill 的 GEMM 密集段混了进来。

按 skill 的要求（"在代表性稳态窗口上做 gap analysis"）改用 **45%–95%** 窗口重跑。该窗口的 decode 性质可由调用次数自证：

| 交叉验证 | 计算 | 结果 |
|---|---|---|
| GDN decode 层数 | 27,264 ÷ 48 层 | **568 步** |
| `_causal_conv1d_update` | 27,264 ÷ 48 | 568 ✓ |
| `kernel_paged_attention_2d` | 9,088 ÷ 16 层（full_attention） | 568 ✓ |

## Decode 步归因（45–95% 窗口，568 步）

窗口总时长 64.27 s（rank0 trace 为空，未重复计数）→ **113 ms/步**，与未 profiling 的 124 ms 一致。

| Kernel | Calls | 每步次数 | Self (s) | Avg (µs) | % Total | 每步耗时 |
|---|---:|---:|---:|---:|---:|---:|
| `ck::kernel_gemm_xdl_cshuffle_v3_multi_d`（INT8 GEMM） | 136,464 | 240 | 38.62 | 283.00 | **60.09%** | **68.0 ms** |
| `kernel_paged_attention_2d.kd`（AITER PA） | 9,088 | 16 | 10.14 | 1116.00 | **15.78%** | 17.9 ms |
| `fused_recurrent_gated_delta_rule_packed_decode`（GDN 线性注意力） | 27,264 | 48 | 9.13 | 335.03 | **14.21%** | 16.1 ms |
| `vllm::dynamic_scaled_int8_quant_kernel` | 145,976 | 257 | 1.00 | 6.83 | 1.55% | 1.8 ms |
| 其余 CK GEMM 实例（2 个） | 9,513 | — | 1.21 | 136/116 | 1.89% | 2.1 ms |
| `_causal_conv1d_update_kernel.kd` | 27,264 | 48 | 0.55 | 20.11 | 0.85% | 1.0 ms |
| 其余（triton 融合 / Cijk / reshape_and_cache 等） | — | — | — | — | <1.5% | <1.7 ms |

**结论：decode 的 60% 花在 INT8 GEMM 上**，注意力（PA 15.8% + GDN 14.2% = 30%）次之，量化只占 1.6%。

**我先前的注意力假设（head_dim=256 拖慢注意力）被否证**：注意力两项合计 30%，且 AITER PA 每步 16 次调用、每次 1116 µs，量级正常。真正的问题是 GEMM。

## 核心量化事实

decode 每步必须把 27.9 GB INT8 权重读一遍，**与 batch 无关**。

| 观测 | 数值 |
|---|---|
| conc 64 时 **GEMM 单项**每步 | **68.0 ms** → 27.9 GB / 0.068 s = **410 GB/s** |
| conc 1 时**整个步** | 33.25 ms（探针 B） |
| 即：同样的权重读取，conc 1 塞得进 33 ms 的一整步，conc 64 光 GEMM 就要 68 ms | **2.0×（对整步）／3.3×（对 GEMM 自身）** |
| 可达带宽下限（探针 A 的 gemv） | 721 GB/s → 27.9 GB / 0.721 = **38.7 ms/步为 GEMM 的现实地板** |
| 纯读上限（探针 A 的 sum） | 1372 GB/s → 20.3 ms/步 |

**所以：M=64 这个 GEMM 形状只跑到 410 GB/s，是它自身现实地板的 57%、纯读上限的 30%。** 而 conc 1（M=1，走 GEMV 类路径）几乎跑满。**M 从 1 到 64 让同一个访存受限的 GEMM 掉了 3.3 倍效率**——这是本模型 conc-64 吞吐被摁在 474 tok/s 的主因。

## 可执行杠杆：`--linear-backend`（**已测，负结论**）

vLLM 0.28 有 `KernelConfig.linear_backend`（`vllm/config/kernel.py:142,217`，默认 `"auto"`），取值包含在 ROCm/gfx90a 上可能有意义的 **`auto` / `aiter` / `triton` / `torch`**（其余是 CUDA 系：cutlass、flashinfer_*、marlin、machete、fbgemm、exon…）。

当前 `auto` 落到的就是 `ck::kernel_gemm_xdl_cshuffle_v3_multi_d`（tile 256×128×128），也就是那 60%。**换 backend 是一个纯 server 参数，风险低、可立刻测。**

> **后续（同日 06:20–07:10 UTC）：已实测完毕，是负结论。** 见 `linear-backend-sweep.md`。
> `auto` / `aiter` / `torch` 落**同一个路径**（稳态中位数精确同为 553.5 tok/s），`triton` 反而慢 21%（435.0）。
> 所以那 60% 不是后端选择问题，而是 **M=64 形状在 CK 与 Triton 下的共同性质**。配置级杠杆至此穷尽。

## 对"要不要重跑 Hyperloom"的最终决策

> **已由 `linear-backend-sweep.md` 判定：不要重跑，且现在是可证的。** `linear_backend` 三个取值同一路径、第四个更差，
> 加上并发空间在 conc 32–128 全平，**配置级杠杆已穷尽**；而 `--no-kernel` 恰好关掉唯一剩下的 kernel 级杠杆，
> 所以原提案 `--no-eval --max-hours 4 --no-kernel` 下不存在通往增益的杠杆，0.00% 是构造性必然。
> 剩余 +35% 需要 kernel 级工作。以下为当时的推理，保留以备对照。

**不建议重跑；建议先做 `linear_backend` 扫描（约 40 分钟）。** 理由：

1. 瓶颈已定位到**一个后端选择问题**（占 decode 60%），而它是**配置级**的——我可以直接扫 4 个取值，成本约 40 分钟；交给 Hyperloom 反而更慢且有不确定性（它的 explore 阶段上次根本没跑到）。
2. 上限要讲清楚：dense 模型每个 decode 步必须读 27.9 GB。即使 GEMM 做到探针实测的 gemv 地板（38.7 ms/步），整步约 84 ms → 约 **640 tok/s**（对比 474，**约 +35%**）。想再往上就得超过"每步读一遍全量权重"这个物理约束——对 dense INT8 不可能。所以 **30% 量级是可达的，数倍不是。**
3. 若扫描出更好的 backend，我们就有了一条**不依赖编排器**的端到端增益；那时再决定是否把它作为起始栈交给 Hyperloom 去 validate。

## 顺带发现的 Magpie 缺陷（与 vLLM 0.28 的不兼容）

1. **TraceLens-inference 启动路径直接崩**：`Magpie/modes/benchmark/tracelens_inference.py:551` 传 `--profiler-config.capture_torch_profiler_dir`，而 vLLM 0.28 的 `ProfilerConfig` 无此字段（pydantic 校验失败），server 在 28 秒内退出：
   ```
   main.py serve: error: argument --profiler-config: 1 validation error for ProfilerConfig
   capture_torch_profiler_dir  Unexpected keyword argument
   ```
   绕法：关掉 `tracelens`，只开 `torch_profiler` 抓 trace，归因走 Magpie 受支持的 **standalone gap analysis**（`magpie benchmark --trace-dir … --top-k N --start-pct/--end-pct`）。本报告即如此完成。
2. **profiled run 收尾挂住**：benchmark 在 05:24:41 已跑完（`Running: 0 reqs`），但 Magpie CLI 与其拉起的 vLLM server 均不退出，33 分钟无任何输出，需人工 kill。
3. 相关配置：vLLM 0.28 的 `ProfilerConfig` 实际字段为 `profiler / torch_profiler_dir / torch_profiler_with_{stack,flops,memory}/ record_shapes / use_gzip / dump_cuda_time_total / capture_torch_profiler / delay|max|warmup|active|wait_iterations`。后者那组**迭代窗口**控制正是抓"稳态 decode 窗口"的正规手段。

## 顺带确认：#1505 补丁**确实生效**

本次直接跑 Magpie（容器 env 里 `HIP_VISIBLE_DEVICES=0,1,…,7` 完好），补丁打出实测诊断：

```
WARNING - Device-mask reconciliation for local run: ROCR_VISIBLE_DEVICES='0' re-indexes the visible set
          to 0..0; reconciled HIP_VISIBLE_DEVICES: '0,1,2,3,4,5,6,7' -> '0'
```

这**补齐了此前悬空的归因**：上一次 baseline 我在 launcher 里 unset 了 HIP，补丁因无矛盾而退化为惰性，故当时只能说"症状类消失"。现在可以确认补丁会触发并把矛盾掩码修正为 `'0'`。已追加到 issue #1505。

## 产物

- `profiling/benchmark_vllm_20260915_051456/torch_trace/`（112 MB torch trace）
- `profiling/gap/gap_analysis/gap_analysis.csv`（全窗口，含完整 kernel 名与 Input Shapes）
- `profiling/gap-decode/gap_analysis/gap_analysis.csv`（**45–95% decode 窗口，本报告的依据**）
- `scripts-local/magpie-profile-conc64.yaml`
