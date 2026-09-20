# GLM-5.3：decode 稀疏注意力的 split-K（本项 7.2x）

> 完整数据见 `hyperloom-mi250x-support-plan.md` 第 12 节与 `quark-int8/sparse_attn_splitk_*.py`。

## 12. decode 稀疏注意力的 split-K（2026-09-21 晚）——本项 7.2x，且**不改内核**

### 定位（全部实测，`quark-int8/sparse_attn_*.py`）

- 生产路径 `_rocm_sparse_attn_prefill_ragged_kernel` 在 **M=1 时 grid 只有 (1, cdiv(128,16)) = 8 个 program**
  （104 CU 的机器上几乎空转）；`BLOCK_D = next_power_of_2(576) = 1024` ⇒ `acc[16,1024]` fp32 = 128 regs/线程（必然溢出），
  且每步 gather 16 行却读 1024 维（真实 576，**浪费 1.78x**）。实测 **1582.6 us/层**，而访存下限仅 **1.4 us/层**。
- **decode 专用内核不能用**：`_rocm_sparse_attn_decode_ragged_triton` 断言 `448 NoPE + 64 RoPE`（DSV4 形状），
  与本模型 `512 + 64` 不兼容；且 gfx90a 上它走的是 "Fallback path for un-tuned architectures"（无 split-K），
  网格尺寸与 prefill 版相同 ⇒ 改调它**不会更快**。

### 做法：把 (query, split) 当作行，一次 launch，再用 LSE 合并

    nq = M*S 行；行序取**查询优先** r = i*S + s ⇒ 源 ragged_indices **零重排**
    新 indptr：starts[i,s] = min(indptr[i] + s*ceil(L_i/S), indptr[i+1])，末项 = indptr[-1]
    q 重复：repeat_interleave(S, 0)；
    合并：w_s = softmax_s(lse_s)，out = Σ_s w_s·out_s，lse = logsumexp_s(lse_s)
          （内核口径 out=acc/l、lse=m+log l ⇒ exp(lse) 即未归一化质量；与 DCP 跨 rank 合并同形）

| 形状 | S=1 | S=2 | S=4 | **S=8** | S=16 |
|---|---|---|---|---|---|
| M=1，本地 1024 行/选 1024 | 799.5 us | 1.90x | 3.83x | **7.18x（111.4 us）** | 5.98x |
| M=1，本地 4096 行/选 2048 | 1583.7 us | — | 3.90x | **7.51x（210.9 us）** | — |
| 索引排序 vs 随机 | — | — | — | 7.15x vs 7.18x（**无关**） | — |

**集成形状是决定性的**：S 次 launch 无效（M=1 时 0.95x，M=4 时 **0.13x**）⇒ 必须一次 launch + 索引按 (q,s) 行序。
（本实现取查询优先行序 ⇒ 连重排都省了。）

### 落点与开关

实现放在 `rocm_sparse_attn_prefill()` 内部的 ragged 分支（**后端零改动**，DCP 合并原样复用）：
`MI250_SPARSE_SPLITK=8`（默认 0 = 关闭，行为与今天逐位一致）、`MI250_SPARSE_SPLITK_MAXM=8`（只在 decode 小 batch 启用）。
缓冲按 (M,S,H,D) 缓存以保证 cudagraph 安全。

### 踩到的坑（留痕）

首次集成测试 lse 完全正确但 **out 全为 0**：低层函数是**返回** out（内部 `empty_like(q)`），我却去读一个从未被写入的
预分配缓冲。修正后对拍：`max|Δlse| ≈ 1e-6`、`max|Δout| = 6.104e-05` —— 后者正是 **bf16 的 1 ulp**
（对 |out|≈1e-2 的值），相对误差 0.19% ≈ bf16 精度 ⇒ 属量化噪声，不是逻辑错误。

### 待做：端到端 A/B（下一步）

`MI250_SPARSE_SPLITK=8` vs `0`：事实召回 6/6 两边都要过；单流 + 并发阶梯对比 TPS。
预期（按 profile 的 36% 占比与 7.2x）：单步 GPU 工作量 186 ms → ~130 ms ⇒ 单流 ~10 → ~14 tok/s（+40%）。
## 13. ⚠️ split-K 端到端 A/B：**负结果**（2026-09-21 04:30）

背靠背两臂（8 卡干净、同配置 `util 0.95 / CG=8 / MBT=2048 / MNS=32 / DCP=8 / 32K`，只差 `MI250_SPARSE_SPLITK`）：

| 指标 | S=0（对照） | S=8 | 变化 |
|---|---|---|---|
| 单流解码（prompt 826） | 9.81 tok/s | 8.65 | **−12%** |
| M=1 聚合 | 7.11 | 6.17 | −13% |
| M=4 | 20.28 | 16.40 | **−19%** |
| M=8 | 37.24 | 31.15 | −16% |
| M=16 | 35.83 | 31.08 | −13% |
| M=32 | 59.68 | 54.72 | −8% |
| **事实召回** | **6/6** | **6/6** | 两臂都正确 ✓ |

对照臂与历史基线吻合（M=1 7.11 vs 7.00、M=32 59.68 vs 59.08）⇒ 平台可比、不是环境漂移。

### 结论与我的判断错在哪

**我在第 12 节预期的"+40%"被实测证伪。** 微基准里那个 7.2x 是真的（内核 799→111 us），但没有兑现到端到端，
反而倒扣 8–19%。最可能的原因（待证，见下）：

1. **这一步是"启动/延迟受限"，不是"内核受限"**：单步已有 1800+ 次内核启动；我的 split-K 实现每层要额外跑
   ~6 个 torch 算子（softmax / mul / sum / logsumexp / 两次 copy）+ 每层一次 q 重复拷贝 ⇒ **每步多出 ~470 次启动**，
   在延迟受限的步里，这些开销很可能超过内核省下的时间。
2. **微基准的 36% 占比可能被 profiler 放大**：profile 是 torch profiler 在 worker 内测的（每内核计时），
   若生产调用实际远低于 861 us/层，那么"砍掉 36%"这个前提就不成立。
3. 合并算子落在 `VLLM_USE_BREAKABLE_CUDAGRAPH=1` 的图里，可能退化成逐层 eager 提交 ⇒ 每层多几次主机侧同步。

### 处置

- 开关保持**默认关闭**（`MI250_SPARSE_SPLITK=0`）⇒ 当前服务配置与之前逐位一致，没有回归；
- 已完成的产物仍有价值：内核级 7.2x 与逐位/1 ulp 级对拍、以及"查询优先行序可零重排"这条结论都成立，
  将来若把合并写进**单个 kernel**（把 ~6 次启动压到 1 次）再评，这条路值得重开；
- **下一步（便宜的判定实验）**：用 worker 内 profiler（`MI250_PROF_WORKER=1`）在 S=8 下再抓一次 decode 窗口，
  与第 11 节的 S=0 归属表逐项对比 ⇒ 直接看"注意力内核时间是否真的掉了、什么变大了"。这能一次性回答上面 1/2/3。
## 14. S=8 下的归属表对照：谜团解开（2026-09-21 05:10）

同一 profiler、同一窗口（ctx=8192、skip=25、8 步），只差 `MI250_SPARSE_SPLITK`：

| kernel | S=0 | **S=8** | 变化 |
|---|---|---|---|
| `_sparse_attn_prefill_ragged_kernel` | 67.174 ms/步（861 us/次） | **11.100 ms/步（142 us/次）** | **−83%** |
| `ncclDevKernel_Generic_4` | 36.267（141 us/次，257 次/步） | **65.457（255 us/次）** | **+80%** |
| `triton_w4a16_gemm_kernel` | 35.260 | 35.262 | 不变 |
| elementwise/copy/cast 调用数 | 1332/步 | **3439/步** | **+2107** |
| **内核总时间** | **186.7 ms/步** | **174.2 ms/步** | −6.7% |

### 三条同时成立的结论（全部实测）

1. **split-K 的 6x 是真拿到了**：内核 861 → 142 us/层，正是微基准承诺的量级（第 12 节的 7.2x 里有一部分被
   头部/尾部开销摊薄了，本表是端到端口径）。
2. **通信被暴露出来**：集合通信原本被那个 861 us 的注意力内核遮住，注意力一快，它就成了关键路径
   （141 → 255 us/次）。**这说明我们一直低估了通信**：即使在 S=0 下它也有 36.3 ms/步（19.4%），
   而 141 us/次对 8 GCD 上的小消息来说异常慢（通常应在 20–40 us 量级）⇒ 值得查 all-reduce 后端
   （vLLM 的 custom/quick all-reduce 是否被禁用）与 RCCL 配置。
3. **我的合并实现太贵**：每步多出 2107 个 elementwise/copy 内核（softmax/mul/sum/logsumexp + 拷贝，
   在 breakable-cudagraph 下还被拆散）⇒ 内核总时间降了 6.7%，墙钟却更长。

### 下一步（按证据排序，都不再是猜）

| # | 动作 | 依据 | 预期 |
|---|---|---|---|
| 1 | 把 split-K 的合并写进**单个 kernel**（把 ~6 次启动/层压到 1 次） | 本表第 3 条（+2107 内核/步） | 让 −6.7% 的内核收益真正落到墙钟 |
| 2 | 攻通信：查 all-reduce 后端与 RCCL 配置，减少 78 层×~3 次集合通信 | 本表第 2 条；S=0 下已占 19.4%，S=8 下 **37.6%** | 单项最大（若 141→40 us，可省 ~25 ms/步） |
| 3 | 之后再评 split-K（1+2 做好后，它才可能正收益） | 三者相加才是完整的账 | — |
## 15. all-reduce 后端：候选里有 4 条快路，实际只拿到 PYNCCL（2026-09-21 05:2x）

### 事实（日志原文，两次独立启动）

    potential backends: [FLASHINFER_PCIE_IPC, FLASHINFER, NCCL_SYMM_MEM, QUICK_REDUCE,
                         AITER_CUSTOM, CUSTOM, SYMM_MEM, PYNCCL]
    Using [PYNCCL] all-reduce backends (in dispatch order) for group tp:0
    Using [PYNCCL] all-reduce backends (in dispatch order) for group dcp:0

⇒ **`tp:0` 与 `dcp:0` 两个组都退到了列表最后一位 PYNCCL（朴素 RCCL 包装）**；
`QUICK_REDUCE`（专为小消息）、`AITER_CUSTOM`、`CUSTOM`、`SYMM_MEM` 全部未被选中。
这与 profile 里"每步 257 次集合通信、单次 141 us（S=8 时 255 us）"互相印证：
小消息路径没走上快路。

### 证伪：不是我们关 AITER 造成的

我最初的假设是"为绕开 gfx90a 上不可用的 AITER MoE，我们把 VLLM_ROCM_USE_AITER 设成 0，连 AR 一起关了"。
**实测证伪**：把 `VLLM_ROCM_USE_AITER=1`（同时保持 `VLLM_ROCM_USE_AITER_MOE=0`）后，
日志里 `tp:0`/`dcp:0` 仍然是 `Using [PYNCCL]` ⇒ AITER 开关不是原因。
（这次启动被用户新规矩"不能抢卡"叫停，但 AR 选择发生在 worker 初始化阶段、日志已落盘 ⇒ 结论有效，
且**没有为此再用一次卡**。）

### 下一步（纯 CPU 可做，不需要卡）

读 vLLM 的 `cuda_communicator.py` / 各 backend 的 `available()` 条件，逐条查清 QUICK_REDUCE / CUSTOM /
AITER_CUSTOM / SYMM_MEM 被拒的原因（大概率是：CUSTOM 的 C++ 内核是 CUDA 专用、QUICK_REDUCE 需要对称内存、
AITER_CUSTOM 需要 gfx942/950 的 aiter AR）。查清后再决定是否有"改一个开关就能走快路"的机会；
任何验证性的起服都要先取得用户许可。
## 16. all-reduce 快路为何全被拒：逐条读源码的判决（2026-09-21 05:4x，纯 CPU）

| 后端 | 在本机不可用的原因（源码级） | 可解? |
|---|---|---|
| `FLASHINFER` / `FLASHINFER_PCIE_IPC` | flashinfer 未安装（CUDA 专用），日志明写 "FlashInfer All Reduce is disabled because flashinfer is not available" | ✗ 结构性 |
| `SYMM_MEM`（torch 对称内存） | 构造条件含 `current_platform.is_cuda()` ⇒ **CUDA 专用** | ✗ 结构性 |
| `QUICK_REDUCE` | 源码注释与检查 `_rocm_arch_available()`：**只支持 ROCm MI300 系列**；禁用原因只打 DEBUG（所以服务日志里看不到） | ✗ 结构性（除非移植） |
| `AITER_CUSTOM` | `rocm_aiter_ops.is_custom_all_reduce_enabled()` = `_AITER_ENABLED and _CUSTOM_ALL_REDUCE_ENABLED`；**实测**把 `VLLM_ROCM_USE_AITER=1` 后仍是 PYNCCL ⇒ 与我们的开关无关 | ✗ 结构性（MI300 系） |
| `CUSTOM` | ROCm 上**不被** P2P 那条挡（该条含 `not current_platform.is_rocm()`），但前面还有 `fully_connected` / size 门；8 GCD 的 XGMI 全连接探测可能判否。日志里**未出现**对应 warning ⇒ 未定论 | **? 唯一可能有戏的一条** |
| `NCCL_SYMM_MEM` | 需 `is_symmetric_memory_enabled()` + world_size 在 `custom_ar_preferred_ranges` | ? 可用 env 试探 |

### ★ 最重要的发现：`dcp:0` 组**注定拿不到任何快路**

`CudaCommunicator.__init__` 开头就按**组名**硬关（`cuda_communicator.py:48-62`）：

    if unique_name.split(":")[0] != "tp":
        use_custom_allreduce = False        # 连带 AITER_CUSTOM / QUICK_REDUCE / CUSTOM
        use_torch_symm_mem = False; use_flashinfer_*= False; use_aiter_allreduce = False

⇒ 只有 `tp:*` 组能享受快路，**`dcp:0` 只能走 PYNCCL（朴素 RCCL）**。
而我们的 DCP 实现**每层都要在 dcp 组上做一次 all-gather + LSE 合并**（78 次/步），
于是"TP all-reduce + DCP all-gather"= 每层两次集合通信，其中一次注定在慢路上。

这与第 11/14 节的实测完全吻合：S=0 时通信 36.3 ms/步（19.4%）、S=8 时 65.5 ms/步（37.6%）、
单次 141→255 us 明显偏慢。

### 由此得到的、**不需要移植任何东西**的方向

1. **减少 DCP 侧通信**：评估 DCP=8 → 4/2 的实际代价（每层通信次数与数据量都降一半/四分之三），
   代价是每 rank 的 KV 变多、KV 池变小 ⇒ 需要实测权衡；
2. **降低 DCP 合并频率**：例如每 N 层合并一次（近似，须验证质量），或把 LSE 合并与 KV 分片收集合并成一次；
3. CUSTOM / NCCL_SYMM_MEM 两条"可能有戏"的路，值得用一次性容器做**判定性探测**（不需要端到端起服）。

⚠️ 以上任何涉及 GPU 的验证都必须先取得用户许可（见 CLAUDE.md 第 1 节新规矩）。
