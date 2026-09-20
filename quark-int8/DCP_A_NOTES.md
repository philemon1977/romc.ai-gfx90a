# DCP 改造 A（LSE）施工笔记（2026-09-20，读码所得，未动 GPU）

真源码位置（bind-mount :ro 进容器，改这里、下次起服生效）：
- 后端：`/home/qiba/ai/patches/gfx90a/ct_w4a16_dsv41_n0918/tree/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`
- 内核：`.../tree/v1/attention/ops/rocm_aiter_mla_sparse.py`
- 参考（CUDA 侧，未打补丁的原版）：容器内 `/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashmla_sparse.py`
  （镜像 `vllm/vllm-openai-rocm:nightly-0918`，vllm `0.3.1.dev85+gdee37d891`）

## 实测调用链（gfx90a Triton 分支）
`forward_mqa`(后端 1035) → `_forward_mla`(831) → `_use_rocm_sparse_triton` 为真(861)
→ `rocm_sparse_attn_prefill(q, kv, indices=None, ragged_indices=paged_kv_indices,
   ragged_indptr=paged_kv_indptr)`(882) → `_rocm_sparse_attn_prefill_ragged_triton`(ops 3196)
→ **单发** `_sparse_attn_prefill_ragged_kernel`(ops 2003)。
⇒ **prefill 与 decode 行都走这一个内核**；indptr 是 [sq+1]（每 query 行一段 ragged）。
⇒ split-KV 的 `_sparse_attn_decode_partial/reduce_kernel`（ops 2438/3077）**不在此链上**
   （它们经 `rocm_sparse_attn_decode`(3977)/`_rocm_sparse_attn_decode_triton`(3773) 进入，
   本后端未调用）。LSE 改造的主战场 = 单发 prefill-ragged 内核。

## A 的具体改法（LSE 每行 float32）
单发内核已有在线 softmax 状态 `m_i/l_i`（ops 2079-2088），收尾在 2090-2105：
- `HAS_ATTN_SINK` 分支已算 `m_final/l_final`；else 分支 `denom=max(l_i,1e-30)`。
改：
1. 内核加参数 `lse_ptr, lse_stride_t, lse_stride_h` + constexpr `WRITE_LSE`；
   收尾 `tl.store(lse_ptr + query_idx*lse_stride_t + head_offsets, m_final + tl.log(l_final))`，
   **空行（l_final==0）写 -inf**（自然对数底；与 CUDA 参考的 log2 底不同，我们的合并自实现，全链统一 ln）。
2. `_rocm_sparse_attn_prefill_ragged_triton`(3196)：`lse=torch.empty(num_queries,num_heads,fp32)`，透传，随 out 一起返回。
3. `rocm_sparse_attn_prefill`(3896) 加 out 参数 `output_lse=None`（对齐 opus/AITER 分支：拿不到 LSE 时要显式报错，不能静默 None）。
4. 后端 `_forward_mla` 897 行 `return output, None` → 返回 lse。
   **注意头维裁剪**：Triton 分支的 output 已过 `get_mla_unpadded_o`（896），lse 同样要按该 helper 语义处理；
   sinks 已在内核内折叠进 m_final/l_final（2090-2096），**不要**再走 1015-1031 的 AITER sink 合并（那是另一分支）。
5. CUDA graph 安全：lse 缓冲在 impl 内**预分配复用**（按 max_num_batched_tokens×heads），不要每次 forward 新建。
6. 若后续为 1M decode 换 split-KV 路径：reduce 内核 3133-3147 处 m_final/l_final 现成，存 LSE 是一行 store。

## 性能预警（验收③要量，别当意外）
单发内核 decode = 每 token 对全上下文一遍在线 softmax。DCP=8 后每 rank 只见 1/8，
但 1M/8=128K 行循环/层 仍是 decode TPS 的主要成本；split-KV（partial+reduce）路径已存在且带 LSE 素材，
1M 阶段若 TPS 不达标，切该路径是第一个杠杆（它同时消费 paged_kv_indices，接口相近）。

## B（索引切分）——设计已定案（17:4x 读上游源码，推翻先前两处误判）
先纠正两个误判：
- 误判①：'indices=None ⇒ attend 全上下文稠密'。**错**。paged_kv_indices 实为
  每 query 行的 **top-2048 选择**（indexer 的 topk_indices_buffer 经
  triton_convert_req_index_to_global_index 换成全局槽位；builder 614-631 的
  generate_sparse_seqlen_triton 给每行有效长度，无效尾部=-1，内核 slot<0 掩掉）。
- 误判②：'先算全局 topk 再切**候选集**'的路径不存在。上游契约（flashmla_sparse.py:1113
  注释原文）："The indexer emits **global** token ids; keep this rank's shard and convert
  to local slots" ⇒ **全局 top-k 由 indexer 产出并保持全局**（indexer K 缓存的
  DCP 布局由平台无关的 v1/attention/backends/mla/indexer.py 管理，DCP-aware 逻辑现成），
  attention 侧只做**过滤+转本地槽**。
我们的改动清单（0002 补丁位，全部在已挂载文件内）：
1. 后端 `supports_dcp = True`（775）+ impl/builder 取 dcp_world_size/dcp_rank（parallel_config 已在 builder __init__ 可用处取）。
2. build() 里把 `triton_convert_req_index_to_global_index` 换成上游
   `triton_filter_and_convert_dcp_index(req_id_per_token, block_table, topk_indices,
   dcp_size, dcp_rank, interleave, BLOCK_SIZE, NUM_TOPK, return_valid_counts=True,
   compact_valid_to_front=True)`：**compaction 到前端** ⇒ 每行只留 ~topk/dcp 个本 rank 槽，
   sparse_seqlen 用其 valid_counts（不再用 generate_sparse_seqlen_triton 的上下文钳制值）。
   （CUDA mixed-batch 用 compact=False 靠 fp8 内核吃 -1；我们 ragged 内核更适合 compact=True，少算 dcp×。）
3. C 合并点放 impl.forward_mqa 上层（拿到 (out,lse) 后 all_gather + ln-LSE 合并）。
4. 钉死配置：dcp_comm_backend=ag_rs（上游唯一验证过）、cp_kv_cache_interleave_size 默认值待 ① 起服日志确认。
风险记录：indexer 在 DCP 下 logits 的行界用 LOCAL bounds（indexer.py:354 注释）⇒
index K 缓存**可能也是分片**的，届时全局 topk 语义 = 各 rank 本地 topk 的并集≠全局 topk。
① 阶段用 128K 针尖直接检验语义正确性；另在起服日志核 index 缓存字节数（分片与否一眼可辨）。

## SMT 实测结论：**对装载无提速**（用户 19:27:55 开 SMT 后的对照）
| 指标 | SMT 关 | SMT 开 |
|---|---|---|
| 每分片耗时 | 5.47–5.55 s | **5.47–5.53 s（无变化）** |
| worker CPU | 8×500–630% ≈ 41 线程 | 8×850–930% ≈ 70 线程 |
| 盘读速（nvme4n1） | 534 MiB/s | **494 MiB/s** |
⇒ CPU 占用涨 70%、吞吐零增益 ⇒ **瓶颈不是线程数**。结合"盘能跑 3.5–4.7 GB/s 却只用 0.5"，
装载卡在**逐张量串行依赖/同步读**这类路径上——加线程/加 SMT 都无效。
深挖需要的下一步（未做）：
1. 单 rank 对照（TP=1 装载同模型，看每 rank 速率是否相同 ⇒ 区分"跨 rank 争用" vs "单流串行"）；
2. `py-spy`/`perf` 采样一个 worker 的栈（需装包 + ptrace 权限）；
3. 对照 `--load-format fastsafetensors` + `FASTSAFETENSORS_UNIFIED_MEM=1`（暂存改统一内存，
   绕开 8 GiB 显存墙）——这是目前唯一还没试过的"换路径"手段。

## 更正：SMT 本来就关着，用户 19:27:55 重启后在 BIOS 打开（实测对照）
- **19:27 之前**：`Thread(s) per core: 1`、`nproc=48`（EPYC 7413 ×2 @24 核）⇒ 装载期 8 个 worker
  各 500–630%、**合计 41/48 线程**、run queue 52–65 ⇒ 当时**确实是 CPU 被打满**。
- **19:27:55 用户重启开 SMT 后**：`smt/active=1`、`CPU(s)=96`、`Thread(s) per core: 2`。同一 mmap
  装载下 8 个 worker 各 **850–930%**、**合计 ~70/96 线程**、run queue 116（>96，仍在排队）。
  ⇒ 装载器**能吃多少线程就吃多少**（torch intra-op 随 nproc 扩），SMT 正对病因；
  对照点：SMT 关 = 666.6 s / 769.1 s（两次实测）；SMT 开 = 见本轮 `Model loading took`。
- 两个过程性教训：
  1. 我 18:5x 报的 "48 核 / SMT off" 在当时是**事实**（非误读）；但 19:31 我把**已停容器**的旧日志
     当成实时进度（该文件 mtime 19:20:52；主机 19:27:55 重启已把容器打成 Exited(255)，
     `restart policy=no` 不自启）⇒ **判读进度前先看 mtime 与 `docker ps`**，别只看 tail 内容。
  2. 守卫的 `FORCE=1` 只用于"同名容器是自己的残骸"；**不要**在守卫之前手打 `docker rm -f`。

## ★ 速度：图模式把单流 TPS 拉高 85%（2026-09-20 21:30 实测）
| 配置（DCP=8 / int4 / TP8 / 32K / MBT=2048） | 单流解码 | 并发4聚合 | KV 池 |
|---|---|---|---|
| eager | 3.59–3.65 tok/s | 12.50 | 577,664 |
| **cudagraph (CG=8)** | **6.38–6.81 tok/s** | **17.69** | 326,016 |
⚠️ 图模式的 capture 池吃显存：KV 池 577K→326K（256K 仍够，1.24×，但余量变薄）。
### 为什么这么低：不是"没用到硬件"，是**形状**（硬件利用率测算）
MoE 激活量：单专家 37.7M 参数 ⇒ int4 18.9 MB + bf16 scale 0.6 MB ≈ **19.5 MB**；
每 token 每层 8 专家 = 156 MB ⇒ TP8 后每 rank 19.5 MB/层 ⇒ 75 层 = **1.46 GB/token/rank**。
6.8 tok/s ⇒ **≈10 GB/s/rank**，MI250 单 GCD HBM ≈1.6 TB/s ⇒ **仅 0.6% 带宽占用**。
⇒ 瓶颈是 **M=1 的 GEMV 形状 + 每 token 1800+ 次小内核启动**：
- MFMA 需要 M≥16 才有吞吐，decode 下每 rank ~1 token ⇒ 矩阵核心闲置；
  MoE 走的还是我们自己的 **GEMV 内核（纯 ALU）**；
- eager 模式下每次启动的固定开销被放大 ⇒ 图模式拿回 ~85%。
本工作区的旁证：同机同族模型 **int8 = 37.57 tok/s vs int4 = 7.16 tok/s**（int8 走原生 int8 矩阵指令）——
差值不在硬件能力，而在数据类型对应的内核路径。
### 提速路线（按实测收益）
1. ✅ **图模式**（+85%，已实测）；
2. **MTP 投机解码**（模型自带 num_nextn=1；launcher 有 SPEC_CONFIG 透传）——进行中；
3. **批处理调优**（并发 4 只拿到 2.6×；让专家 GEMV 变 M>1 的 GEMM 才能真正吃矩阵核心）；
4. **MoE 内核重做基准**（并行会话那次因"权重/scale 布局与生产不同构"而无效）+ DCP 通信开销隔离。

## ★ GEMV 为什么吃不到 HBM 带宽：ALU 受限（2026-09-20 读内核 + 算术账）

> ⚠️ **本节结论已被文末「v3：真因是 scale 取用」推翻（同日晚）**：实测只有 0.9 Tops/s（fp32 峰值 4%），"ALU 被吃光"的算术账与实测不符；四条处方里只有第 1 条（scale 提到 group 级）命中真因，但机制不是"少算 ALU"而是"少发 16 倍取指"。保留原文以留痕。
内核形状（`moe_gemv/mi250_moe_gemv_gs.py`）：
```
grid = (num_valid, N // 128)   BLOCK_N=128  BLOCK_K=max(group,128)=128  num_warps=4
每个 K 步加载 [128, 64] uint8 = 8 KB 权重；每元素：
  nib = (wb >> 4j) & 0xF - 8 → .to(fp32) × xj.to(fp32) × sc.to(fp32) → tl.sum(axis=1)
  ⇒ ≈4–6 flops + 2 次类型转换 / 权重元素 ⇒ **8–12 flops/字节**
```
**MI250 单 GCD 的算力/带宽平衡点**：FP32 ≈ 110 CU × 64 lane × 2 × 1.7 GHz ≈ **24 TFLOP/s**；
HBM ≈ **1.6 TB/s** ⇒ 平衡点 ≈ **15 flops/字节**。我们压在算力侧 ⇒ 实测 110 GB/s（HBM 的 ~7%）。
**推论（重要，别走弯路）**：这不是"没喂饱带宽"，是"ALU 被吃光"。
- 加并发也救不了：top-8/256 专家，batch=256 时每专家 M≈8，仍喂不动 MFMA（要 M≥16）；
- M=1 的 GEMV 形状下，矩阵核心**原理上**用不上（MFMA tile 16×16 ⇒ 15/16 浪费）。
**唯一路径 = 砍每元素的 ALU**（四处，都在同一文件）：
1. **scale 提升到 group 级**：每 32 元素乘一次，而非每元素乘一次（现写法 `sc` 逐元素取）⇒ −25~30% ALU；
2. **bf16 中间量**：先算 `xj*sc`（bf16）再累加，减少 int→fp32 转换次数；
3. `BLOCK_N=256` + `num_warps=8`（当前 128/4）提高在飞请求数；
4. gemm2（down）也接管 + **融合逆置换**（patch 注释说它 ~38 us 太小、被逆置换吃掉收益 ⇒ 需先把逆置换做成单次 kernel）。
**诚实预期**：MoE ≈ 单步 40%（补丁自述实测）⇒ 优化 1.5–2.5× on MoE ⇒ **单流 +20~40%（6.8 → 8.5–9.5 tok/s）**；
**单流 30 tok/s 不现实**（要 4.4×，除非换 int8 权重路线）。
已排的验证：`moe_gemv_bench.py`（真实 checkpoint 布局：gs=32、uint8 [E,4096,3072]+bf16[E,4096,192]、
down uint8[E,6144,1024]+bf16[E,6144,64]），扫 (BLOCK_N, BLOCK_K, warps) 并报有效 GB/s。

## 速度现状（2026-09-20 21:16 实测，DCP=8 / int4 / TP8 / eager / 32K / MBT=2048）
| 场景 | 实测 |
|---|---|
| 单流解码（prompt 410–1642） | **3.59–3.65 tok/s**，TTFT 0.83–1.86 s |
| 首个请求（202 token） | 1.39 tok/s（warmup 未计入稳态） |
| 并发 4 | **12.50 tok/s 聚合**（单流均值 3.33） |
| KV 池 | 577,664 token @32K（17.6×），本轮 0 次 OOM |
历史对照：**DCP=1 同权重同卡 eager = 3.72–3.77 tok/s** ⇒ DCP=8 目前仅差 ~3%（待正式头对头）。
**显存余量教训**：util 0.97 + MBT=4096 时，16K prompt 预填充会 OOM（"Tried to allocate 576 MiB, 0 free"，
引擎随之死）⇒ 把 prefill 批降到 **2048** 后余量足够（KV 池反而增到 577K），16K prompt 不再 OOM。
提速杠杆（按顺序实测中）：① 图模式（ENFORCE_EAGER=0 + CG）② MTP spec（模型自带 num_nextn=1，
launcher 有 SPEC_CONFIG 透传）③ DCP=0 对照 ④ MoE GEMV 内核（并行会话方向）。

## 🎉 里程碑：DCP 端到端正确（2026-09-20 20:46，DCP=8 @32K 事实召回 **6/6**）
修复 = 0005（ROCm indexer 的本地→全局 top-K 合并）+ 0006（逐行长度本地化）。
证据：`logs/...` 中事实召回 6/6，且 `' Paris. Distance from Paris to Lyon is'`、
`'北京，美国首都华盛顿，英国首都'`、`'2? - 简书'` 与 DCP=1 基线**逐字相同**。
调试数字（修复后，3-token prompt）：rank0 valid=3 / rank1 valid=2 / rank2 valid=1 /
rank3-7 valid=0（3 个 token 只用到 3 个 rank ✓）；**lens ≥ 0**（负长度消失）；
`valid>len 行数=0`；行内 -1 空洞按设计存在。
两个中间坑（都已修）：
1. op 内不能调 `get_current_vllm_config()`（请求路径在 breakable-cudagraph 上下文里，
   实测 `AssertionError: Current vLLM config is not set`）⇒ 改为后端 `__init__` 写入
   ops 模块的模块级 `_DCP_TOPK_CTX`（`set_dcp_topk_ctx`）；
2. 新增 `@triton.jit` 内核必须插在**装饰器之前**——插到后面会"双重装饰 + 原函数丢装饰"，
   报 `TypeError: ... expected, got JITFunction`（实测踩过）。

## ★★ DCP 乱码总根因（2026-09-20 20:15 调试取证，两条独立缺陷）
调试开关：`MI250_DCP_DEBUG=1`（已在 launcher 透传），在已挂载文件里打 3 次即停。
实测数字（3-token prompt，DCP=8，block_size=16，interleave=1）：
```
rank0: valid sum=3  lens[min/max]=-1/1  valid>len 行数=2
rank1..7: valid sum=0  本rank用到的槽=0  out_absmean=0  lse finite=0/192（全 -inf）
```
### 缺陷①：ROCm indexer 完全没有 DCP 感知（**主因**）
- 证据：`grep -c dcp v1/attention/ops/rocm_aiter_mla_sparse.py` = **0**；
  `forward_hip` 调 `torch.ops.vllm.rocm_aiter_sparse_attn_indexer(...)` 的参数里**没有任何 dcp_\***；
  只有 CUDA 的 `forward_cuda` 才把 `dcp_rank/dcp_world_size/cp_kv_cache_interleave_size` 传给 op。
- 后果：DCP 下 index-K 也是分片的 ⇒ ROCm indexer 产出的是**本 rank 的本地 id**，
  而"本地 top-K → 全局 top-K"的合并只存在于 CUDA 路径 ⇒ 注意力侧把本地 id 当全局 id 过滤
  ⇒ 除 rank0 外全部 valid=0 ⇒ 内核读到 0 个槽 ⇒ out=0、lse=-inf ⇒ 输出确定性乱码
  （也解释了"换 indexer logits 内核输出逐字相同"与"DCP=2 乱码不同于 DCP=8"）。
- **我上一轮的 0002 补丁改错了文件**：`_merge_dcp_topk_global` 在 `sparse_attn_indexer.py`
  （CUDA/XPU 路径），我们这条 ROCm 路径不经过它。⇒ 需 **0005**：在
  `rocm_aiter_sparse_attn_indexer`（ops 文件）的 prefill 与 decode 两个出 topk 的位置，
  调用"候选交换 + 全局 top-K"（可复用 0002 已写好的 pack/torch-topk 实现），
  dcp 参数直接从 `get_current_vllm_config().parallel_config` + `get_dcp_group()` 取，**不改 op 签名**。
### 缺陷②：per-row 长度本地化用错了公式（会产生负长度）
- `generate_sparse_seqlen_kernel` 用 `local_seq_len - query_len + offset`：本 rank 只有 1 个
  token 而 query_len=3 ⇒ context_start = -2 ⇒ **行长度为负**、`paged_kv_indptr` 非单调，
  还出现 `valid > len`（有效槽被截掉）。
- 正解：对**每个 query 行的全局前缀长度**单独做本地化
  `local_ctx_i = get_dcp_local_seq_lens(prefix_len_i)`，其中
  `prefix_len_i = (seq_len - query_len) + offset + 1`，再 clamp 到 topk。⇒ **0006**。
- 顺带解释：这条也影响 rank0（它 valid=3 但 lens 有负数/截断）。

## ROCm indexer DCP 缺口## ★ fastsafetensors 成功配方（2026-09-20 19:52 实测，装载 666–769 s → **108.66 s**，起服 121 s）
```
--load-format fastsafetensors
MI250_FST_MAX_BATCH_MB=2560          # 必须 ≥ 最大单张量（本模型 embed/lm_head = 1.77 GiB）
FASTSAFETENSORS_ODIRECT=1            # 绕页缓存（= llama.cpp dio 的等价物）
```
三个坑，逐个实测排除（每次失败都在 ~2 分钟内，代价低）：
1. 默认（无切块）⇒ **OOM**：整文件批次（2.85–4.5 GiB，1–2 个在飞）+ 权重 52.95 GiB > 64 GiB；
   而 `--gpu-memory-utilization` **管不到装载期**（它只约束 KV 池，实测 0.97→0.94 无效）；
2. `device_memory_budget` ⇒ **BudgetInfeasibleError**：该参数是"**整个模型**的设备预算"，
   不是单批上限（52.95 GiB ≤ 4 GiB 直接判不可行）；
3. `max_batch_bytes=1024` ⇒ **ValueError: smaller than tensor 'lm_head.weight'** ⇒ 取 2560 MiB。
- vLLM 未透出切块参数 ⇒ 打了个小补丁 `weight_utils.py`（新增 `MI250_FST_MAX_BATCH_MB` /
  `MI250_FST_DEVICE_BUDGET_MB`，未设时与上游逐字一致）；已挂进 launcher（mount 第 50 行 + env 80/81 行）。
- `VLLM_FASTSAFETENSORS_QUEUE_SIZE` 默认已是 0（无缓冲）⇒ 不是可调项。
- vLLM 自带的 safetensors/mmap 路径**没有任何 DIO 旋钮**；dio 只能经 FST 的 `ODIRECT`。
- 已默认固化进 `scripts_local/glm_dcp_boot.sh`（`FAST_LOAD=0` 回退 mmap）。

## 归因更正：DCP 乱码**不是** indexer 路径造成（上一轮结论被证伪）
19:52 轮：`DSV41_IDX_AITER_KERNEL=1` **确已生效**（日志 = `自研内核(SHUFFLE)`，"不可信"ERROR 消失），
**但事实召回仍 0/6，输出与上一轮逐字相同**（`',7款识人之'` / `'0##'` / `'Comments11111'` …）
⇒ 同一个**确定性**错误，indexer 选择不是根因。
正在做对照（`scripts_local/glm_dcp0_control.sh`）：**DCP=0 @32K + 同一套补丁 + FST 装载**跑同一批事实召回：
- 6/6 ⇒ 乱码由 DCP 路径（filter/convert、本地 seq_lens、层合并 三者之一）引起；
- 同样乱码 ⇒ 是我们补丁本身在任何配置下都坏，与 DCP 无关。
这正是 2 分钟装载的红利：对照实验从 14 分钟一轮变成随便做。

## DCP 输出乱码的根因（2026-09-20 19:14 实测，**配置级，不是代码 bug**）
现象：DCP=8@256K 起服成功、请求不再崩（`seq_lens` 修复生效），但**事实召回 0/6**（DCP=1 时 6/6）、
GSM8K 吐 `clips clips clips...` 乱码；1-token 微探针也乱（结构性问题，不是权重微差）。
根因（服务端日志自证）：
```
[DSV41] indexer logits 路径 = 上游 torch 回退(不可信) (block_size=16, mode='默认')
ERROR [DSV41] fp8_paged_mqa_logits_torch 在 block_size=16 的 SHUFFLE 缓存上结果不可信
```
**DCP 让 KV cache 的 block_size 变成 16** ⇒ indexer 的 decode logits 默认走"上游 torch 回退"，
而它按行主序读 SHUFFLE 页缓存，本机结果不可信（我们自己的 ops 里早写了这条 ERROR，launcher 注释里
也写明："未设/0 = 上游 torch 回退——不可信；1 = 自研 gfx90a 内核（按 SHUFFLE 反解，已过三方对拍，支持 BS=16）"）。
⇒ top-K 选错 ⇒ 注意力attend 到错误的 token ⇒ 乱码。
**修法（零代码）**：`DSV41_IDX_AITER_KERNEL=1`。已**机制化**：
`scripts_local/glm_dcp_boot.sh` 现在默认导出该值（`${DSV41_IDX_AITER_KERNEL:-1}`），
DCP 起服不可能再忘。19:17 已带此值重启（container 2f97297c73b3）。

## 起服守卫的两条实测教训
1. **守卫有效**：`docker rm -f` 后显存未落净时，gate 3 直接拒绝起服（"这些 GCD 仍被占用"），
   而不是硬起导致同伴 OOM ✓ 机制按设计工作。
2. **但不要手动预备 `docker rm -f`**：我 19:16 在守卫之前手打了一次 rm，把正在跑的验收电池
   最后一格（parity 探针）打断成 connection refused。正确姿势：只写 `FORCE=1 bash glm_dcp_boot.sh`，
   删旧容器由 launcher 在**守卫通过之后**做。

## 装载速度：实测结论（2026-09-20，用户提问触发）
1. **不是盘慢**：同一块 CM6，单流 O_DIRECT 4 MiB **3.5 GB/s**、8 路并行 **4.67 GB/s**；
   而 vLLM 的 safetensors(mmap) 装载只有 **0.52 GB/s**（402 GB / 769 s）。
2. **是 CPU 受限**：装载期 8 个 worker 各 **500–630% CPU**、合计 **41/48 核**、run queue 52–65；
   机器为 **EPYC 7413 ×2 = 48 物理核，SMT 关闭**。⇒ 缓存/hipFile 类 I/O 手段**对错了资源**；
   唯一对得上诊断的硬件手段是 **BIOS 开 SMT**（预期 +20~40%，需整机重启，尚未做）。
3. **fastsafetensors 在本模型上顶显存**：文件级 GPU 批次 + 我们 **52.95 GiB/rank** 权重超预算
   （OOM 现场：要 1.77 GiB、只剩 1.62 GiB）。对比：Ornith int4 **43.12 GiB/die** 可行
   （该臂实测 **336 s → 131 s，2.6×**）；DSV4.1 **61.64 GiB/rank** 据此否决 FST。
   我们的中间档 → 用 **util 0.94** 补 ~1.9 GiB 余量重试（KV 池仍须 ≥ 2.83 GiB 才保得住 256K）。
4. **AIS/hipFile 在 serve 模式判死**（非推断）：`docs/MI250X-hipFile-AIS-部署手册.md` +
   `config/ais.env` 记载——离线可跑（216 s / 43.12 GiB per die），但 api_server 进程树里
   `cuFileBufRegister 5014` 必崩且抛 Exception 绕过回退；上游又硬编码 `nogds = pg.size() > 1`
   ⇒ TP=8 本就不给 GDS。现成可用组合 = fastsafetensors **pinned bounce（nogds）**。
5. 已撤回的中间方案：**400 GB 按 rank 分片副本**（脚本 `scripts_local/glm_shard_convert.sh` 留档）。
   撤回理由：设备总读量不变（页缓存原本就在去重），CPU 受限下"少读字节"未必省时；
   跑到 ~40% 装载后中止、产物已清理（`/mnt/stripe-3mix-3t2` 空间已回收）。

## ② 的显存账（2026-09-20 定量，用本轮 17:38 实测反算校准）——**bf16 latent 到不了 1M**
每 token 每 rank 的真实成本（config: 78 层、kv_lora 512+rope 64、21 个 full indexer、index_head_dim 128）：
- latent：78 × 576 × 2 B = **89,856 B = 87.8 KiB/token**
- indexer K 缓存：21 × (128 + 4) B = **2,772 B = 2.71 KiB/token**
- 合计 **92,628 B = 90.5 KiB/token** ⟵ 与施工单里"实测 90.5 KB/token/rank"**逐位吻合**（独立复算确认）。

1M token 的三种情形 vs 实测可用量：
| 情形 | 需要/rank | 实测/预算 |
|---|---|---|
| DCP=8，latent + indexer 都切 8 份 | **11.31 GiB** | util 0.97 预算 11.22 GiB（差 0.09） |
| DCP=8，latent 切 8、indexer 复制 | **13.68 GiB** | 任何 util ≤0.99 都装不下（0.99 也才 12.50） |
| 实测本次（32K/bf16/DCP=1）：KV 池 93,776 token | 8.09 GiB 池 | 预算 11.22 ⇒ **差额 ~3.1 GiB 是非 KV 开销（激活/workspace）** |

**结论（重要）**：
1. util 0.99 只多给 1.3 GiB，而缺口是"3.1 GiB 激活开销 + 0.09~2.5 GiB 分片差"
   ⇒ **bf16 latent 在 gfx90a 上达不到 1,048,576 token**（乐观上限 ~75–90 万 token）。
2. 所以 ② 若要坚持 1M，**必须上量化 latent（fp8/int8，latent 砍半 ⇒ 5.5 GiB/rank，宽裕）**；
   而 gfx90a 的 ragged Triton 路径目前看起来是 **bf16-only**（`_load_fp8_ds_mla_*` 那套是
   gfx950 专属；`_use_rocm_sparse_triton` 在 gfx90a 上无条件返回 True，不区分 dtype）
   ⇒ 这是一块**新增内核工作**，必须显式记入工作量与退路评估。
3. 施工单的退路（128–256K）则**非常宽裕**：256K 只需 2.83 GiB、512K 只需 5.66 GiB（DCP=8）。
4. 上面每条都可由"裸测"直接验证：`--max-model-len 1048576`（不开 DCP）起服，读
   `GPU KV cache size` / `Available KV cache memory`（脚本：`scripts_local/glm_dcp_boot.sh`
   配 `DCP_SIZE=0 SKIP_PATCH_CHECK=1`）。**② 的可行性在动 B/C 之前就能钉死。**

## 0003 修正（2026-09-20 首跑实测）：**合并由层做，后端只返回 (out, lse)**
首跑 DCP=8@256K 报 `Decode Context Parallelism (DCP) requires attention implementations to
return the softmax LSE during decode` ⇒ 缺的是**能力声明**，不是实现。
读镜像源码定案（`model_executor/layers/attention/mla_attention.py`）：
```python
attn_out, lse = self.impl.forward_mqa(mqa_q, kv_cache, attn_metadata, self)
if self.impl.dcp_world_size > 1:
    assert lse is not None
    attn_out = self.dcp_manager.combine(attn_out, lse, seq_lens=..., query_start_loc=...)
```
⇒ 后端**只负责把 (out, lse) 返回出去**；自己再合并就是**双重合并、静默算错**。
我原先在 0003 里写的 `_merge_dcp_partial` 已撤销（补丁重生成，hunks 4→3，99 行）。
正确的三件套是：
1. `supports_dcp = True`
2. `can_return_lse_for_decode = True`（基类默认 False；`__new__` 据此置
   `need_to_return_lse_for_decode`）
3. `lse_base_on_e = True`（我们的内核是**自然对数底** `m_final + log(l_final)`；基类注释明确
   警告底数写错会静默污染跨分片分母 ⇒ 显式写出，不靠默认值）
另外两条上游已备好、不用我们写的能力：
- `MLADCPManager.correct_attn_out / mask_dcp_empty_shards_`：空分片行（本 rank 无 slot）的修正；
- `dcp_manager.query_gather`（decode 查询聚合）与 `init_kv_gather`（**prefill 跨 rank KV gather**
  ——这正是我之前标为"B 的开放问题"的那件事，上游已解）。
（对照：flash_attn / flashinfer 是**后端自持** `dcp_combine` 并自行调用；稀疏 MLA 路径相反，
由层合并。两条路线不同，别互相套用。）

## ② 的实测显存（DCP=8 首跑，util 0.97，max-model-len 262144）
- `Available KV cache memory: 5.35 GiB`、`GPU KV cache size: 484,096 tokens`、
  **Maximum concurrency for 262,144 tokens per request: 1.85x** ⇒ **256K 目标达成**（>1x）。
- 与 DCP=1 对比：DCP=1 时同样 util 下可用 8.09 GiB（93,776 token @32K）。
  ⇒ **DCP=8 自身额外吃掉约 2.7 GiB**（query gather / KV gather workspace / 合并缓冲）。
- 容量外推（92,628/8 = 11,578 B/token/rank）：util 0.97 ⇒ 484K；0.98 ⇒ ~505K；0.99 ⇒ ~520K。
  ⇒ **512K 差一点点，1M 仍必须量化 latent**（与"bf16 上限 75–90 万"的粗算相比，实测更紧）。
- 调优阶梯（256K 稳了之后再走）：① util 0.97→0.98/0.99；② 缩小 max-num-seqs/batched-tokens 省
  workspace；③ fp8/int8 latent（砍半 ⇒ 1M 才有戏）。

## 0003 修正前的原始设计（考古，勿按此实施）

## C/0003 蓝图（attention 侧 DCP，锚点与写法已定，待 ⓪ 门放行后生成）
改动全在 `v1/attention/backends/mla/rocm_aiter_mla_sparse.py`（我们已挂载，无新文件、无需改挂载表）：
1. `ROCMAiterMLASparseImpl.supports_dcp = False → True`（775）。
2. `__init__`（819 已取 `vllm_config = get_current_vllm_config()`）加三个字段：
   `self.dcp_world_size = parallel_config.decode_context_parallel_size`、
   `self.dcp_rank = get_dcp_group().rank_in_group if >1 else 0`、
   `self.cp_interleave = parallel_config.cp_kv_cache_interleave_size`。
3. `forward_mqa` 的转换调用（1069-1077）改成 dispatch：
   ```python
   if self.dcp_world_size > 1:
       from vllm.v1.attention.backends.mla.sparse_utils import triton_filter_and_convert_dcp_index
       slots, valid_counts = triton_filter_and_convert_dcp_index(
           attn_metadata.req_id_per_token, attn_metadata.block_table, topk_indices,
           dcp_size=self.dcp_world_size, dcp_rank=self.dcp_rank,
           cp_kv_cache_interleave_size=self.cp_interleave,
           BLOCK_SIZE=attn_metadata.block_size,
           NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
           compact_valid_to_front=True, return_valid_counts=True)
       fetch_id_to_ragged_triton(slots, attn_metadata.paged_kv_indptr,
                                attn_metadata.paged_kv_indices, attn_metadata.topk_tokens)
   else:
       triton_convert_req_index_to_global_index(...)   # 原样不动
   ```
   两个关键点：
   - **`fetch_id_to_ragged_triton`（本文件 321，现成）**就是"稠密 [rows,topk] → ragged 扁缓冲"的打包器，
     不用新写内核；compaction 后行内 [valid_count, len) 全是 -1，我们的 sparse 内核本来就掩 slot<0。
   - **`paged_kv_indptr` 不用改**：仍由 `generate_sparse_seqlen_triton` 按上下文长度 min(ctx,topk) 生成，
     行内多出来的位置是 -1 空洞，被内核掩掉。⇒ 只需换转换函数，行长度契约不变（这是最小改动路径）。
4. C 合并（依赖 0001 返回的 lse），放在 `forward_mqa` 返回前：
   `all_gather(out) + all_gather(lse)` → 归约到 [world, T, ...] → `m = lse.max(0)`、
   `w = exp(lse - m)`（-inf → 0）、`out = Σ w·out_fp32 / Σ w`（空行输出 0）。
   与 ops 里 `_sparse_attn_decode_reduce_kernel`(3133-3184) 的数学同构，fp32 累加。
   注意：每层都要 gather（78 层）⇒ 这是 ③ 必须量的 DCP 开销。
5. 补丁顺序：**0001 → 0003**（同一文件不同 hunk，但 0003 需在 0001 之后生成/应用）。
   生成时以 `dcp_patches/work/backend.py`（0001 的输出）为基线，避免上下文冲突。

## B 补：indexer 侧「本地 top-K → 全局 top-K」已找到上游实现，移植件已写（0002）
读码结论（`model_executor/layers/sparse_attn_indexer.py`，我们已挂载的文件）：
- 纯 DCP 路径：prefill 只用**本 rank 分片** gather（`cp_gather_indexer_k_quant_cache` +
  `chunk.local_cu_seq_lens`）⇒ logits 是本地的 ⇒ 本地 top-K ⇒ **必须合并**（调用点 630，
  条件 `deinterleave_idx is None`）；decode 同理（调用点 767，条件 `global_seq_lens is not None`）。
- PCP+DCP 路径：`dcp_gather_kv_rows` 全量 gather ⇒ top-K 已全局 ⇒ 跳过合并。
- 合并的精确性论证（上游 docstring）：全局 top-K 里的 token 必然在其所属 rank 的本地 top-K 内
  （全局排在它前面的 ≤ topk-1 个，落在同 rank 的更少）⇒ 只交换各 rank 候选即等价于 all-gather 全矩阵，
  通信量 `world*topk` 而非整行 logits。
- **gfx90a 缺口只有一处**：稳定的 top-k 选择器是 CuteDSL（radix-select，u64 key、hist_bins=2048、
  要求候选数是 512 的倍数）；pack 本身是 Triton，可直接搬。
⇒ 已生成 `dcp_patches/0002_gfx90a_indexer_dcp_topk.patch`（搬 Triton pack + torch.topk 复现语义，
CUDA 侧不变）。已知差异：完全并列时次序由实现决定（真实 fp32 logits 下概率极低），
若 parity 出现抖动再换 u64 单调 key 的稳定选择（备选 0004）。

## B 早期草稿（保留考古，勿再按此实施）
- DCP 下 KV 分片：每 rank 的 `paged_kv_indices/indptr` 由 core builder 给出的是**本 rank 持有的分片**，
  行长度用 `dcp_local_seq_lens`（CUDA 参考 503-523：`get_dcp_local_seq_lens`、
  `dcp_local_seq_lens_cpu_upper_bound`；`triton_filter_and_convert_dcp_index` 做全局索引→本 rank 过滤转换）。
- 本路径当前 indices=None（全上下文稠密 ragged）⇒ 切分=前缀按 rank 交错，无需 topk 过滤；
  **若**将来上 DSA top-k（GLM 有 indexer），必须先全局 top-k 再过滤到本 rank 分片（计划 B 的"别搞反"）。
- 开放问题：chunked prefill 的第 N 块要看 0..N-1 块的 latent —— 那些块已写进**别的 rank** 的分片。
  CUDA 侧靠 PCP+DCP 的 `pcp_dcp_kv_gather`（且限 fp8_ds_mla，参考 392-396）。我们是 bf16 latent，
  要么自实现 prefill 期 all-gather 分片 KV（只影响 TTFT），要么首版限制 prefill 策略。**未定案，先做 DCP=2 解码正确性。**

## C（合并）要点
- partial(out,lse) → `get_dcp_group().all_gather` → fp32 log-sum-exp 合并；数学与 reduce 内核 3133-3184 完全同构
  （m=max(lse_r)，out=Σ exp(lse_r−m)·out_r / Σ exp(lse_r−m)）。
- **rank 分片为空的行 lse=-inf**：先取 m 再 exp，全 -inf 时输出 0（对齐内核 3184 的 `where`）。
- 参考 386-389：DCP 只验证过 `dcp_comm_backend='ag_rs'`；`supports_dcp_with_varlen` 要求
  `cp_kv_cache_interleave_size==1`（参考 312-317）。这两个配置要在 launcher 里钉死。

## ⓪ 实测暴露的真阻塞（比 DCP 更靠前）：indexer logits OOM（已修，17:38 复跑中）
- 现象：17:18 轮服务器 READY 后，首个 ~8K prompt 请求即 **8 rank 同刻 torch.OutOfMemoryError
  （"Tried to allocate 2.00 GiB"，free 仅 818 MiB）→ 引擎全灭 → Py_Exit Segfault**。
  尺子：quark-int8/logs/glm53_verify32k_0920_1718.log + /home/qiba/ai/logs/glm53/server-8121-20260920-171823.log:353+。
- 根因（归属明确）：gfx90a 无 aiter/flydsl ⇒ indexer prefill 永远走 `fp8_mqa_logits_torch`（ops 973），
  一次性物化 `[H=64,M,N]` fp32 中间量；而 `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256` 的预算与
  `split_indexer_prefill_chunks` 的分块只约束 **输出 [M,N]**，管不到 [H,M,N] ⇒ 64× 超预算。
  4096×4096 chunk 的 einsum bf16 输出恰好 2.00 GiB，与报错严丝合缝。
- 修复（并行会话落码 17:3x，本会话审查通过）：按 M 分块，`m_chunk=budget/(H·N·4)`，
  逐元素数值不变、无 .item() 同步；文件 ops 966-1007（★ 注释）。
  本会话独立推导过等价的按 H 累加方案（峰值 O(M·N)≈512MiB），两者选一即可，**保持现状不叠改**。
  遗留：长上下文下 m_chunk 退化为个位数 ⇒ python 循环启动次数 ∝ M/m_chunk，1M 阶段要换
  **融合 Triton logits+top-k**（对齐 _dsv41_paged_mqa_logits_gfx90a 的做法，decode 已有、prefill 缺）。
- **撞车事故记录（透明留痕）**：17:38:41 对方会话起服验证该修复；17:38:43 本会话（不知对方已起）
  的 glm_verify32k.sh 因 launcher `docker rm -f glm53-int4`（固定容器名）把对方 2 秒新的容器顶掉。
  根因与 CLAUDE.md §1 记载的旧事故同源：**launcher 无条件按固定名 rm**。
  处置：保留单服继续装载（同代码同配置），锁 quark-int8/logs/.verify32k.lock/info 留了致歉与交接说明；
  待办：给 launcher 加「名字已被占则拒绝并提示」的硬门（_guard.sh 同款思路），杜绝复发。

## 现场状态（本轮 17:31）
- `glm53_verify32k_0920_1718` 服务器装载中（17:18 起，READY≈17:31）。
- **已修 `agent_bench.py` 两个 bug**（否则本轮 needle 又白跑）：
  ① `doc()` `unit[0] % tag` TypeError（16:27 那轮的崩因，line 52→`unit % tag`）；
  ② needle 原来不判命中——`stream()` 现在收集文本，needle 打印 `答案=... [HIT/MISS]`。
- 16:27 轮已证：事实召回 6/6、GSM8K 5/6（1 题算错，非路径问题）；needle 当时未跑成。
- 门控：⓪ 16K 针尖 HIT + graph/eager 对照过了才动内核。

## 🎉 v3：MoE GEMV 从 60 GB/s 提到 275 GB/s（2026-09-20 晚，推翻「ALU 受限」）

### 定位过程（都是实测，quark-int8/moe_gemv_diag.py，单 GCD，真实 checkpoint 布局 [E,N,K/2]）

| 实验 | 结果 | 说明 |
|---|---|---|
| 全量内核（v1/v2） | 1674.5 us / **60.1 GB/s** | 基线 |
| **去掉 scale 乘法** | **378.2 us / 266.2 GB/s** | **4.4x！瓶颈就是 scale** |
| 去掉 nibble 提取 | 1663.9 us / 60.5 GB/s | nibble 解码几乎免费 |
| 纯 load（同访存模式） | 112-162 us / **620-898 GB/s** | 访存模式本身能到 900 GB/s |
| 二维累加器（v2） | 1.01x | **证伪**「跨 lane 归约是主因」 |
| 算术账 | 0.9 Tops/s = fp32 峰值 4% | **证伪**「ALU 受限」 |
| n_regs/spills | 73 / 0 | 无寄存器溢出 |

**真因**：v1 的 scale 下标是逐元素的 kk // GROUP（kk = k0 + 2c + j）。这个**非线性下标**让
Triton 无法向量化，每个 k 步退化成 [BLOCK_N, BLOCK_K//2] 次 **2 字节 gather**（BLOCK_K=64 时
2048 次取指，其中只有 2 个不同地址）⇒ **取指数被放大 ~16 倍**。

**v3 修法**（moe_gemv/mi250_moe_gemv_v3.py，已内联进 mi250_moe_gemv_gs.py）：
BLOCK_K = G_PER_STEP x GROUP，**三维累加器** [BLOCK_N, G_PER_STEP, GROUP//2]，k 循环内完全不碰
scale；每步收尾把第三维归约掉、只乘一次 [BLOCK_N, G_PER_STEP] 的 scale 切片（同一行内这 G 个
值在内存里连续，可向量化）。开关：MI250_MOE_GEMV_KERNEL=v3（默认）/ v1。

### 微基准（对拍 fp32 参考**均为 0.14%**，与 v1/v2 同级 ⇒ 精度不变）

| 形状 | v1 | v3 最优 | 提速 |
|---|---|---|---|
| 全 N：gemm1 K=6144 N=4096 | 1694.5 us | **412.3 us**（BN=64 G=8 w=4） | **4.1x** |
| 全 N：gemm2 K=2048 N=6144 | 891.1 us | **208.5 us** | **4.3x** |
| **生产分片**：gemm1 K=6144 **N_local=512** | 391.4 us | **45.2 us**（**BN=16 G=8 w=1**） | **8.7x** |
| **生产分片**：gemm2 K=2048 N_local=768 | 132.9 us | **26.5 us** | **5.0x** |

生产分片形状由 MI250_MOE_GEMV_DEBUG=1 的 DUMP 行**实测确认**（不是推测）：
A(8,6144) C(8,8,512) B(256,512,3072) Bs(256,512,192) pairs=64 top_k=8 ⇒ N_local=512、gs=32。
⇒ **网格只有 8x8=64 个 program（104 CU 的机器严重欠占用）**，所以「小 BLOCK_N 换更多 program」
比「每 program 干得多」重要得多（全 N 那份调优的 BN=64 在分片形状下是错的，差 2 倍）。
生产里 v1 是 **524 us/层 ⇒ 78 层 = 40.9 ms/token = 147 ms 步长的 28%** —— 与下面实测的端到端
+45% 吻合（不是旧笔记按 ALU 推算的 3.6%）。

### 端到端（同脚本同配置：DCP=8/TP8/int4/32K/CG=8/MBT=2048/MNS=32，KV 池 326,016 可对照）

| 并发 | v1 内核 | **v3** | 变化 |
|---|---|---|---|
| 1 | 5.31 | **6.92 → 7.00** | +30% |
| 4 | 13.82 | **20.43 → 21.73** | +48% |
| 8 | ~20 | **40.64** | **~2x** |
| 16 | 27.25 | **34.45**（另一轮 27.81，噪声大） | +26% |
| 32 | 33.80 | **58.59 → 59.08** | **+73%** |

单流解码 **6.38-6.81 → 9.60-10.71 tok/s**；事实召回 **6/6**（回答与验收基线逐字一致）。
日志：logs/conc_v3_0920.log、logs/conc_v3tuned_both_0920.log。

### 同批的负结论（别重走）

- **DCP=0（完全不开 DCP）单流并不更快**：7.66 vs DCP=8 的 9.60 tok/s，且 KV 池掉到 64,176
  ⇒ 无理由放弃 DCP=8。日志 logs/conc_v3_dcp0_0920.log。
- **调优本身在生产里看不出来**：MoE GEMV 从 v1 的 28% 降到 ~5% 后，BN=64→16 的收益淹没在
  探针噪声里（6.92 vs 7.00）⇒ 探针精度不足以分辨 <5% 的改动，别拿它当判据。
- **gemm2 是否真被接管还没有证据**：DEBUG 打印落在 takeover % 200 == 1（恒为奇数次调用 ⇒ 恒为
  gemm1），所以那些 apply_w=False 行**不能**证明 gemm2 没接管。要按 kind 分别计数重打，或在微基准里
  直接单测 invoke_gemv_wna16 的第二次调用。

### 下一步（按已测量的结构）

瓶颈已不在 MoE：单步 147 ms 里 MoE GEMV 只剩 ~5.5 ms。剩下 95% 用 rocprofv3 kernel-trace 定位
（scripts_local/glm_prof.sh KTRACE=1 + analyze_ktrace_window.py，窗口由 marker kernel 界定）。
备选：MTP 投机解码（模型自带 num_nextn=1，launcher 有 SPEC_CONFIG 透传，尚未落地）。
## MTP 投机解码：实测**净亏**（2026-09-21 00:10，DCP=8 / TP8 / 32K / CG=8）

做法：launcher 的 SPEC_CONFIG 透传，`SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":1}'`。
vLLM 能正确解析（日志 `Resolved architecture: DeepseekV32MTPModel`，checkpoint 里 layers.78 的
2339 个张量齐全，`num_nextn_predict_layers=1`）。

**装载期先撞 OOM**（`Tried to allocate 1.77 GiB`，已分配 58.64 GiB、仅剩 950 MiB）——draft 层又叠了
一份 embed/lm_head 级的内存。修法（已验证有效）：`FASTSAFETENSORS_UNIFIED_MEM=1`
+ `MI250_FST_MAX_BATCH_MB=1920`（暂存放统一内存 + 收小切块）。

| 并发 | 无 MTP（v3） | **MTP** | 变化 |
|---|---|---|---|
| 单流解码 | 9.60–10.71 | **7.63–8.41** | **-20%** |
| 1 | 7.00 | 5.66 | -19% |
| 4 | 21.73 | 16.89 | -22% |
| 8 | 40.64 | **17.45** | **-57%** |
| 16 | 34.45 | 29.68 | -14% |
| KV 池 | 326,016 | **173,184** | 腰斩 |

投机**本身是有效的**：`SpecDecoding metrics: Mean acceptance length 1.18 → 1.67`（即每步平均多出
0.18–0.67 个 token）。但每步要多跑**一整个 MTP 层**——它不是 1/78 的代价，因为它要再走一次完整
的稀疏注意力 + indexer，成本与主干的注意力同级。净亏，故**不启用**。

**顺带得到的旁证（有价值）**：一个额外注意力层就能吃掉单流 ~20% ⇒ 注意力/indexer 在单步里是
大头，这与"MoE GEMV 优化完只剩 5%"互相印证。下一步定位应直奔注意力/indexer 路径。

日志：logs/conc_mtp2_0920.log；服务端 /home/qiba/ai/logs/glm53/server-8122-20260920-235937.log。

### 还试过但没走通的定位手段（留痕）

- `rocprofv3 --kernel-trace`（图模式与 eager 各一次）都在**装载结束、引擎初始化**处卡死：
  GPU 0% 占用、worker 232% CPU 空转、rocprofv3 自己的 signal handler 也挂住 ⇒ 疑与 RCCL 初始化冲突。
- `torch.profiler` 在 driver 进程里**看不到任何 kernel**（只有 8.8 us 的 hipDeviceSynchronize），
  因为 vLLM v1 把模型跑在独立 worker 进程 ⇒ 要 profile 必须注入 worker 进程（sitecustomize 钩子）。
## 🔍 剩下 95% 在哪：worker 内 profiler 的 decode 窗口归属（2026-09-21 02:30）

### 手段（两条路都试过，只有这条通）
- ❌ `rocprofv3 --kernel-trace`：图模式与 eager 两次都在**装载结束、引擎初始化**处死锁（GPU 0%、worker
  232% CPU 空转、连它自己的 signal handler 都挂住）⇒ 疑与 RCCL/多进程初始化冲突。
- ❌ driver 进程内的 `torch.profiler`：只有 8.8 µs 的 hipDeviceSynchronize —— vLLM v1 把模型跑在**独立
  worker 进程**里 ⇒ 什么都看不到。
- ✅ **worker 内注入**：`moe_gemv/sitecustomize.py` 里加 env 门控钩子（`MI250_PROF_WORKER=1`），
  在 worker 进程内挂钩 `GPUModelRunner.execute_model`，抓第 skip..skip+steps 步，各 rank 写
  `worker_rank<N>.{txt,json}`。配套 `quark-int8/analyze_worker_prof.py` 聚合、
  `scripts_local/glm_prof.sh` 起一次性容器。默认关闭，对生产零影响。

### decode 窗口（ctx=8192、M=1、skip=25、8 步；8 个 rank 的 GPU 工作量 185–190 ms/步，高度一致）

| kernel | ms/步 | 占比 | 调用/步 | 单次 |
|---|---|---|---|---|
| `_sparse_attn_prefill_ragged_kernel`（DSA 稀疏注意力） | **67.2** | **36.0%** | **78**（每层 1 次） | 861 µs |
| `ncclDevKernel_Generic_4`（TP all-reduce + DCP 通信） | **36.3** | **19.4%** | 257 | 141 µs |
| `triton_w4a16_gemm_kernel`（上游 WNA16，非专家 MoE 部分） | **35.3** | **18.9%** | 261 | 135 µs |
| hipBLASLt `Cijk_...MT64x16x16` | 11.7 | 6.2% | 75 | 155 µs |
| **我们的 `_gemv_moe_v3`** | **3.3** | **1.8%** | 75 | 45 µs |
| MoE 路由/topk/align 等杂项合计 | 11.0 | 5.9% | 642 | |

⇒ **MoE GEMV 这条线已经吃完**（1.8%）。下一个 4 倍只能在**稀疏注意力内核**、**每层集合通信**、
**非专家的 W4A16 GEMM** 三处找。

### ⚠️ 更正留痕（我先说错了一次）
第一次 profile 抓的 8 步窗口里混进了 **prefill 分块**（MBT=2048 切 8192 的 prompt ⇒ 4 个分块步），
于是我把 `_sparse_attn_prefill_ragged_kernel` 占 52.7% 读成"decode 在跑 prefill 形状的内核"，
并据此怀疑是 DCP 行数 bug。**这个推论是错的**：
- 窗口墙钟 160–169 ms/步、且含 `vllm::unified_mla_attention_with_output`（635 ms）——那条在纯 decode
  窗口里**根本不出现**，说明它属于 prefill；
- 换 skip=25 的**纯 decode 窗口**后，该内核仍有 **78 次/步**（每层一次），但这是 DCP 下 DSA 的
  decode 稀疏注意力内核本身（名字里的 prefill 是历史命名），不是"把 prefill 塞进 decode"。
- ctx=512 的插桩跑里它只被调 6 次/整个 run ⇒ **疑似只有上下文超过 index_topk(2048) 才走稀疏路径**
  （短上下文走 dense）。这条阈值假设还没专门验证，别当结论用。

### 由此得到的可用判断
- **单流 TPS 与上下文强相关**：ctx≈800 时实测 10 tok/s（≈98 ms/步），ctx=8192 时 ≈4 tok/s（≈250 ms/步）
  —— 差异主要来自这个稀疏注意力内核（以及随之增加的通信）。报告单流 TPS 必须写明上下文长度。
- `ncclDevKernel` 257 次/步、单次 141 µs 偏慢（78 层的 all-reduce + DCP 通信都在里面），是第二个可下手处。
- `triton_w4a16_gemm_kernel` 261 次/步（≈3.3 次/层）是**非专家**的 int4 GEMM（共享专家/稠密层/lm_head），
  同样是 M=1 形状 —— 与 v3 修的是同一类病，值得按同样思路过一遍。
### ⚠️ 发现：仓库里的 .patch 比线上树旧（待收尾）

验证方法：把 `dcp_patches/base/ops.py.orig` 依次打上 0001/0005（唯一两个触及 ops 的补丁），与线上树 diff。
结果：**只有一处差异**，在 `rocm_aiter_sparse_attn_indexer` 附近——树上是模块级 `_DCP_TOPK_CTX`
（`set_dcp_topk_ctx` 由后端 `__init__` 写入），而补丁生成出来的是在 op 里调
`get_current_vllm_config()` 的旧版本。

来源：本会话早些时候为修 `AssertionError: Current vLLM config is not set`（op 在 breakable-cudagraph
上下文里执行）时**只改了树、没回写 make_patch*.py** ⇒ 克隆仓库按 README 打补丁会得到**会报错的旧版**。
收尾动作（未做，留给下一步）：把该 hunk 写回生成器并重跑 `make_patch*.py`，或直接新增 0007 补丁。
在此之前，**以线上树为准**（`/home/qiba/ai/patches/gfx90a/ct_w4a16_dsv41_n0918/tree/`）。

另：本次为定位插在 ops 文件里的 `_sparse_dbg` 探针已**全部移除**并再次 diff 确认（除上述已知 hunk 外零差异）；
诊断手段记在这里备查（env 门控、默认关闭）：在 `rocm_sparse_attn_prefill` / `rocm_sparse_attn_decode`
入口打一行 `traceback.extract_stack()` 的调用链 + 形状，用 `MI250_SPARSE_DBG=1` 打开、`MI250_SPARSE_DBG_N` 限次数。
