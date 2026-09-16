# `linear_backend` 扫描：负结论 —— 配置级杠杆已穷尽

日期 2026-09-15 06:20–07:10 UTC。目的：验证 profiling 指出的"decode 60% 在 INT8 GEMM"是否可通过换后端解决。

脚本 `scripts-local/sweep_linear_backend.sh`；汇总 `scripts-local/summarize_linear_backend.py`；产物 `linear-backend-sweep/`。

## 动机

`profiling-decode-attribution.md` 的 45–95% 稳态窗口显示，conc 64 的 decode 步里 **60.09%** 花在
`ck::kernel_gemm_xdl_cshuffle_v3_multi_d`（283 µs/次，240 次/步），而该 kernel 只跑到 **410 GB/s**，
是它自身现实地板（探针 A 的 gemv 721 GB/s → 38.7 ms/步）的 57%。
vLLM 0.28 有 `KernelConfig.linear_backend`（`--linear-backend`，`arg_utils.py:1603`，默认 `auto`），
取值中在 gfx90a 上有意义的是 `auto / aiter / triton / torch`。这是唯一一个直接对着该 60% 的**配置级**杠杆。

## 方法

每个取值都要**重启 server**（后端选择在初始化时定死）。负载与探针 B 的 conc 64 点一致：
ISL/OSL 1024、128 prompts、`--max-concurrency 64`、`--ignore-eos`、TP=1 单 GCD、同 shim 与同 server args。

bench 的 `output_throughput` 含启动爬坡（首个完成时间 auto 2:27 / aiter 2:29 / triton 3:10），
所以另从 server 日志取**满载采样中位数**（`Running >= 32`）做同口径对比。

## 结果

| backend | bench output tok/s | **稳态中位数 tok/s** | 稳态采样数 | TPOT mean |
|---|---:|---:|---:|---:|
| `auto`（默认） | 439.48 | **553.5** | 28 | 134.58 ms |
| `aiter` | 437.24 | **553.5** | 28 | 134.33 ms |
| `torch` | 437.26 | **553.5** | 28 | 134.30 ms |
| `triton` | **341.15** | **435.0** | 36 | 172.11 ms |

（steady 中位数高于 bench 均值是因为后者含爬坡；这一偏差在四个取值上同构，故相对比较有效。）

## 结论：负结论，且很干净

1. **`auto` / `aiter` / `torch` 是同一个路径**：三者稳态中位数精确相同（553.5），bench 值差 0.5% 以内。
   这印证了机制上的判断——**AITER 的 a8w8 INT8 linear 本身就建立在 CK `xdl_cshuffle_v3` 之上**，
   所以 `VLLM_ROCM_USE_AITER_LINEAR=1` 与 `auto`/`torch` 落到同一 kernel。
2. **`triton` 明显更差**（-21%，TPOT 172 vs 134 ms）。Triton 的 int8 GEMM 在 gfx90a 上不如 CK。
3. **没有任何取值带来增益。** 那 60% 的 GEMM 成本**不是后端选择问题**，而是 **M=64 这个形状在 CK 与 Triton 两种实现下的共同性质**。

## 这意味着什么

对 `Qwen3.8-27B-ABLITERATED-W8A8-gdnint8`，**配置级杠杆到此全部量完并穷尽**：

| 杠杆空间 | 测量结果 | 出处 |
|---|---|---|
| 并发 / 调度 | conc 32→128 全在 468.8–491.7 平台区（+4.9%） | `probe-concurrency-saturation.md` |
| 线性层后端 | 3 个取值同一路径；第 4 个慢 21% | 本报告 |
| 精度门 `RUN_EVAL` | 只影响预算，不影响吞吐 | `shared/orchestration-budget-and-teardown.md` |
| 掩码 / 启动类 | 已修（#1505 补丁实测生效） | issue #1505 |

**剩余余量是 kernel 级的**：GEMM 每步 68 ms，而它自己的地板是 38.7 ms（探针实测 gemv 速率）。
若能补上，整步 113 → 约 84 ms，conc 64 约 **640 tok/s（+35%）**。再往上要突破"每步读一遍 27.9 GB 权重"，
对 dense INT8 不可能——**所以 30% 量级可达，数倍不是。**

## 对"要不要重跑 Hyperloom"的最终判定

**不要重跑，且现在是可证的，不再是估计：**

原提案是 `--no-eval --max-hours 4 --no-kernel`。其中：
- `--no-eval` 省的是**预算**，不是吞吐；
- `--max-hours 4` 让 `integrate` 的 7800s cap 放得下，但**没有可集成的杠杆**；
- `--no-kernel` **恰好关掉唯一还剩的杠杆**（kernel 级）。

即：**该配置下不存在通往增益的杠杆**，0.00% 是构造性的必然。要继续追那 +35%，需要的是 kernel 级工作
（为 M=64 的 INT8 GEMM 找/写更好的 tile 配置），属另一个项目：Magpie `analyze`/`compare` 在隔离形状上迭代，
或打开 `--no-kernel` 让 Hyperloom 去碰 kernel——但 gfx90a 上 AITER 的 ASM int8 kernel 全部不可移植
（移植矩阵：482 个 int8 kernel 被挡），等于要从 Triton/CK 从头写。

## 产物

- `linear-backend-sweep/sweep.csv`、`lb-{auto,aiter,triton,torch}.json`、`bench-*.log`、`server-*.log`
- `scripts-local/sweep_linear_backend.sh`、`scripts-local/summarize_linear_backend.py`
