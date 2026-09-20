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
