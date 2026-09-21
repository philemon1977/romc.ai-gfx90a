# decode 配置栈消融（2026-09-21：DCP / split-K / QuickReduce / 自研 indexer 内核）

**尺子（先读，否则数字会被误用）**：`quark-int8/qr_tps_probe.py` 量的是**请求级吞吐**
（含 prefill 与首 token 延迟，ISL≈700、OSL=128、temp0、流式，token 数取 usage.completion_tokens）。
它**不可**与报告里 9.60–10.71 tok/s（纯 decode 口径）互比；只有本表内各臂可横比。
服务配置：TP8、max-model-len 32768、util 0.97、enforce-eager=1、镜像 glm53-int4-gfx90a-0918(-qr)。

| 臂 | 配置 | conc=1 | conc=8 | conc=32 | 召回 |
|---|---|---|---|---|---|
| A | stock 镜像，默认栈 | 4.38 | 28.15 | 74.77 | 6/6 |
| A′ | -qr 镜像，三条 QR env 全不设 | 4.18 | 26.64 | 73.13 | 6/6 |
| S2 | + split-K(S=8, MAXM=32) + indexer 内核 | 3.71 | 27.93 | 107.68 | 6/6 |
| **S2a** | + **仅** indexer 内核 | **4.54** | **32.51** | **123.78** | 6/6 |
| S3 | + indexer 内核 + DCP=8（bf16 KV） | 3.18 | 22.83 | 89.12 | 6/6 |

## 一、DSV41_IDX_AITER_KERNEL=1：今天唯一显著正向，而它默认是关的

A′→S2a：conc1 +8.6%、conc8 +22.0%、**conc32 +69.3%**（73.13→123.78），召回 6/6。
launcher 注释自陈：未设时走上游 torch 回退，按行主序读 SHUFFLE 页缓存、**本机结果不可信**；
`=1` 是已三方对拍的自研 gfx90a 内核。**当前默认值同时是更慢与更不可信的那一支。**

## 二、split-K：三档全负，默认保持 0（上午的结论被强化）

S2 vs S2a（唯一差异是 split-K）：conc1 −18.3%、conc8 −14.1%、conc32 −13.0%。
`MI250_SPARSE_SPLITK_MAXM` 默认 8 ⇒ conc=32 时 split-K 根本不进场，上午的高并发是盲区；
放开到 32 仍为负。内核 7.2× 换不回端到端，因为每步 elementwise/copy 从 1332 涨到 3439。

## 三、QuickReduce：不是没调好，是算不过账（决定性负结论）

四个开 QR 的臂全部死在启动显存门：`Free memory 54.9/63.98 GiB < desired 0.97 (62.06 GiB)`；
不开 QR 的臂同一时刻是 62.9–63.03 GiB。差值 ~9 GiB/卡，且
`VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=16` 压不动它（失败数字一模一样）⇒ 属
`ops.init_custom_qr()` 的固定内部分配。本模型 KV 总预算仅 8.17 GiB ⇒ 与 QR 互斥。
前人笔记的三条 env 在宽权重模型上不完整，第四条 env 也不解决内存问题。

## 四、DCP=8：省的不是显存，是每 token 的显存单价

| | A′（DCP=1） | S3（DCP=8） |
|---|---|---|
| 权重/rank | 50.84 GiB | 52.95 GiB（+2.11：MLA 投影按 rank 复制） |
| KV 预算 | 8.17 GiB | 5.89 GiB |
| KV 容量 | 94,016 tokens（2.87× @32k） | **534,784 tokens（16.32× @32k）** |
| 每 token KV 单价 | 93.4 KiB | **11.5 KiB（降到 1/8.1）** |
| 短上下文吞吐 | 123.78（S2a） | 89.12（−28%） |

**1M 的算术**：DCP=8 + bf16 KV = 534,784 < 1,048,576 ⇒ 单条 1M 装不下；再叠 fp8 KV（单价再半）
≈ 1.05M ⇒ 勉强过线。**fp8 KV × DCP=8 是 1M 的必要条件组合。** fp8 KV 臂已确认被配置层接受，
但按用户要求在装载阶段停止，**KV 实测数未取**（恢复后的第一臂）。

## 五、两条 launcher 地雷（已修，且都属于默认值即地雷）

1. 枚举型 env 用 `-e NAME=${NAME:-}` 注入**空串** ≠ 未设置：vLLM 抛
   `Invalid value for VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` ⇒ worker 必死。改为非空才注入。
2. `MAX_CUDAGRAPH_CAPTURE_SIZE` 默认 0 + `ENFORCE_EAGER=0` ⇒ vLLM 断言拒绝：
   **eager=0 这条路此前从未走通**。已加自动取 MAX_NUM_SEQS 的防御。

## 六、方法论（这次差点记错账）

首轮按理论最优一臂开五杠杆，得到 conc32 +47.2% 并记在 split-K 头上；单杠杆消融后发现那 +47%
全属 indexer 内核，而 split-K 实际 −13%。**一臂一杠杆是硬规矩**；混测只能探天花板，不能记收益。
第二个教训：ready() 只轮询 HTTP 会为秒死的容器空等 45 分钟。fail-fast（查容器 State + 连续两次
判死 + 先存日志再删容器）把单次失败反馈压到约 100 秒 —— 今天的四次二分就是在 15 分钟内完成的。

## 七、复现（全部不依赖 GPU 之外的东西）

```bash
source quark-int8/gpu_gate.sh && gate 8121              # 门：三条单测，不抢卡不腾地方
bash quark-int8/stack_probe.sh <臂名> KEY=VALUE ...      # 等卡→起服→召回+TPS→停服→追加结果表
TPS_ISL_MULT=24 bash quark-int8/stack_probe.sh LONG ...  # 长档（ISL≈17k），DCP 必须在长档判
python3 quark-int8/verify_patches.py                     # 补丁队列/树自洽（含 0007-0009）
```
机器可读结果：同目录 `stack_results_20260921.jsonl`（`quark-int8/logs/` 是 gitignore 的）。
