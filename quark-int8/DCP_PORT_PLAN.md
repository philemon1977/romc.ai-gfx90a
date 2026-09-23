# GLM-5.3 上 MI250X 的 DCP 移植施工单（2026-09-20 立）

目标：让单条 1M(1048576) token 上下文成为可能。
为什么必须 DCP：MLA latent KV 在 TP 下每 rank 复制，实测 **90.5 KB/token/rank**
⇒ 1M 需 90.46 GiB/rank > 单卡 64 GiB ⇒ **不切分 KV 就物理不可能**。
DCP 把 KV 沿序列切到 N 个 rank：1M ⇒ 90.46/N GiB/rank；N=8 ⇒ 11.3 GiB/rank ✓ 可行。

## 已确认的入口（本轮实测/读码所得，下一轮从此处继续）
1. 配置：`vllm/config/model.py:1444` 起 —— 校验 `decode_context_parallel_size`；
   `max_dcp_size = tensor_parallel_size // total_num_kv_heads` ⇒ MLA(1 个 KV head) 可到 8 ✓
   CLI：`--decode-context-parallel-size N`（见 `engine/arg_utils.py`，下一轮确认确切名）
2. 后端：`v1/attention/backends/mla/rocm_aiter_mla_sparse.py`（我们已放行 gfx90a 走 Triton 分支 ✓）
3. 我方内核：`v1/attention/ops/rocm_aiter_mla_sparse.py`
   - `rocm_sparse_attn_prefill(..., ragged_indices=paged_kv_indices, ragged_indptr=...)` ✓ 生产在用
   - `_rocm_sparse_attn_decode_ragged_triton` / `_rocm_sparse_attn_decode_triton`
4. 参考实现（CUDA 侧已有 DCP 合并逻辑，可对照移植）：
   `v1/attention/backends/mla/flashmla_sparse.py`（含 `get_dcp_group()`、LSE 合并、
   `"DCP for FlashMLA sparse is only supported on the mixed-batch fp8 path"` 的限制）

## 已知阻塞点（必须解决，否则 DCP 无法工作）
- **A. LSE 缺失**：我们选中的 Triton 分支 `_forward_mla` 目前 `return output, None` ✗
  ⇒ DCP 合并需要**每个 token 的 LSE**（flashmla_sparse 注释明确：separate 路径只给 decode 返 LSE）。
  要做：让内核额外写 LSE（每行 float32），并在 `_forward_mla` 返回它。
  施工笔记（调用链实测 + 行号锚点 + B/C 开放问题）见 `quark-int8/DCP_A_NOTES.md`（2026-09-20）。
  B 的 indexer 侧（本地 top-K → 全局 top-K）上游已有实现、只缺 CuteDSL 选择器 ⇒
  `quark-int8/dcp_patches/0002_gfx90a_indexer_dcp_topk.patch` 已备好（含起 DCP 的 6 项"必须翻"清单）。
  A+B+C 三件补丁（0001 LSE / 0002 indexer DCP / 0003 attention 分片+合并）**已全部离线生成并链式验证**
  （顺序应用→编译→回滚逐字节还原），只等 ⓪ 门放行上树；① 的判据改成 `dcp_patches/dcp_parity_probe.py`。
  A 的补丁已离线生成待放行：`quark-int8/dcp_patches/0001_gfx90a_sparse_mla_lse.patch`（README 有尺子顺序）。
- **B. 索引切分**：`paged_kv_indices/indptr` 需按 DCP rank 分片（每个 rank 只负责序列的一段）。
  要做：在 metadata builder 内按 dcp_rank 切 [start,end)，且 topk 选择结果需可切分
  （注意：DSA 的 top-2048 是**全局**选择 ⇒ 先算全局 topk，再按 rank 切分**选中集合**，
   而不是切分候选集；这一步是设计核心，别搞反）。
- **C. 合并**：各 rank 出 partial output + LSE ⇒ 用 `get_dcp_group().all_gather` + LSE 加权合并
  （照抄 flashmla_sparse 的 merge，dtype fp32 累加）。
- **D. 与 SWA/compressor 无关**：GLM 无 compressor（DSV4 专属）⇒ 只搬 DCP 合并，别把 compressor 依赖拖进来。

## 目标修订（2026-09-20 用户指示）：**先落 256K，成功后再按实测显存调优**
- 为什么不是 1M 起步：每 token 92,628 B（latent 87.8 KiB + indexer 2.71 KiB，见 DCP_A_NOTES.md
  的显存账），实测每 rank KV 池 8.09 GiB（util 0.97）⇒ 全局容量 ≈ 池 × DCP / 92,628 B。
  **DCP=2 → ~187K（装不下 256K）；DCP=4 → ~387K；DCP=8 → ~750K ✓**（512K 也在射程内）。
  ⇒ 256K 用 **DCP=8**（原判据①的 DCP=2 只到 ~187K，按此修订）；bf16 latent 上限 ~75–90 万 token
  ⇒ 1M 仍需量化 latent（新增内核工作），留作后续。
- 调优顺序（256K 成功之后）：① 读起服日志的真实 `GPU KV cache size` 与 `Available KV cache memory`；
  ② 提 util（0.97→0.98/0.99）换容量，量质量与 TPS 是否退化；③ 视需要再评估 fp8/int8 latent（砍半）。
- 执行记录：用户授权停服重启 ⇒ 停掉并行会话 18:04 的性能跑（当时仍在装载，未损失测量），
  三件补丁上树，**端口 8122 起 DCP=8 + max-model-len 262144**（独立端口，避免对方探针静默测错配置）。

## 判据修订（2026-09-20 评审，**原判据保留在下方不删**）
- **⓪ 的针尖尺寸标注错误（测具问题，非模型问题）**：`agent_bench.py` 的 doc() 按
  "英文≈3.6 字符/token" 估算，实测为 **2.03 字符/token**（8K 目标→4504、16K 目标→9232，
  两点比值 0.5634/0.5635）。⇒ 17:38 轮的"16K 针尖"实为 **9.2K**，"8K×3 轮"实为 4.5K×3 轮；
  若照此推到 1M，实到仅 ~563K。测具已修：`quark-int8/agent_bench2.py`（回读 usage 自校准）。
- **针尖 HIT/MISS 不能作为"稀疏路径/DCP 正确性"判据**：config 实测 `index_topk=2048`，
  9.2K 上下文里每条 query 只对 2048 个被选中的 token 做注意力（78% 的位置进不了）。
  该轮填充是**同一句重复 ~1000 次**，与 top-k 选择共振 ⇒ 针尖可能整段未被选中，
  此时 MISS 反映的是 DSA 选择，而不是我们的 Triton 内核或 DCP 合并。失配的判据会误导施工。
- **①③ 改判据为 DCP 等价性（parity）**：同一 prompt、同一权重下，
  `DCP=1` 与 `DCP=2/8` 的输出必须一致（贪心 token 逐位相同，或 logits 相对误差 < 阈值），
  并叠加 KV cache 容量与显存断言。DCP 合并写错表现为"与 DCP=1 不一致"，不是"针尖 miss"。
- **针尖保留为 ④ 质量回归**，但必须：填充逐条唯一（避免与 top-k 共振）、针尖用易复述数字码、
  出**命中率曲线**（2K/4K/8K/16K/32K，一次起服跑完）而非单点 HIT/MISS。
- **归因仪器**：`DSV41_IDX_DUMP=1`（自带 ops 1475-1503）dump 每层 logits/topk_indices 到容器
  `/tmp/idx_dump`，配合 `quark-int8/dcp_patches/analyze_idx_dump.py` 判定"针尖是否在选择集内"，
  把 MISS 归因到"选择"或"注意力"。此前仓库内无任何脚本消费该 dump。
- 17:38 轮仍成立的结论：indexer OOM 修复 **PASS**（同尺寸请求此前必崩，本轮全程 0 OOM）、
  事实召回 6/6 与 GSM8K 5/6 与 16:27 轮逐题一致（无回归）、前缀缓存 TTFT 比 0.092、
  解码 3.76 tok/s 与既有权衡基线 3.71 tok/s 一致（非回归）。

## 验收判据（顺序执行，原版）
0. 先量 **graph vs eager** 与 **16K/128K 针尖**（正在跑的 verify 日志：`logs/glm53_verify32k_*.log`）
   —— 稀疏路径正确性是 DCP 的前提 ✗ 未过就别做 DCP
1. `--decode-context-parallel-size 2` 起服成功且 **128K 针尖仍正确**（DCP 合并正确性的最小验证）
2. `--decode-context-parallel-size 8` + `--max-model-len 1048576` 起服：
   `GPU KV cache size` 应 ≥ 1,048,576 token；`Available KV cache memory` 每 rank 需 > 11.3 GiB
   （权重 50.8 GiB + KV 11.3 GiB + 开销 ⇒ util 0.97 下勉强够，必要时 g128 scale 省 4 GiB/rank）
   ⚠️ **2026-09-20 定量复核：这条按 bf16 latent 大概率达不到，判据需先重估再执行。**
   复算依据（见 `DCP_A_NOTES.md` 的"② 的显存账"）：每 token 真实成本 92,628 B（latent 87.8 KiB
   + 21 个 full indexer 的 2.71 KiB），1M 在 DCP=8 下需 11.31 GiB（indexer 也切）或 13.68 GiB
   （indexer 复制）；而 17:38 实测 32K 时的 KV 池只有 8.09 GiB —— 预算 11.22 GiB 与实际池之差
   ~3.1 GiB 是非 KV 开销（激活/workspace）。util 顶到 0.99 也只多 1.3 GiB ⇒ **bf16 latent 的
   乐观上限约 75–90 万 token**。结论：1M 若必须，需 **量化 latent（fp8/int8，latent 砍半）**，
   而 gfx90a 的 ragged Triton 路径目前是 bf16-only（fp8 装载器是 gfx950 专属）⇒ 这是新增内核工作量。
   **执行前先做裸测**（`DCP_SIZE=0 SKIP_PATCH_CHECK=1 bash scripts_local/glm_dcp_boot.sh`，
   max_model_len=1048576）读回 `GPU KV cache size`/`Available KV cache memory` 定案。
3. 端到端：1M 针尖命中 + TTFT/TPS 与无 DCP 对照（DCP 会引入 all_gather 开销，必须量）
4. 质量回归：事实召回 6/6 不退化（现值：GLM int4 = 6/6 ✓ GSM8K ≥4/4 ✓）

## 风险与退路
- vLLM 对 **稀疏 + DCP** 的支持在 CUDA 侧也有限制（见上面的 only-supported 报错）⇒ 可能要先
  实现我们自己的合并（不依赖上游那条限制路径）。
- 退路（若 DCP 成本失控）：把上下文目标降到 128–256K（`fp8 latent` + `g128 scale` 可达），
  agent 会话场景多数够用；1M 留作后续。
- 工作量粗估：A(0.5d) + B(1d) + C(0.5d) + 调试/起服循环(1–2d，每轮装载 12.5 分钟) ⇒ **3–4 天**

## 其它待办（别丢）
- MoE tile 调优基准**当前无效**（权重/scale 布局与生产不同构：需 `int32 [N,K/8] + bf16 scale`）；
  已修成'失败不当结论'，待重做基准后再谈 +TPS。
- `scripts_local/glm_watch.sh` 仍引用 /tmp（待清理）；`perf_bench.py` 的 TTFT 是非流式近似，
  我之前加的 `stream_options` 在那是死代码（要么删、要么改真流式）。
