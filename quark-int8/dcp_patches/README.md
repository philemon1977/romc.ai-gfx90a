# dcp_patches —— DCP 移植的离线补丁队列（一次一个改动上树）

## 0001_gfx90a_sparse_mla_lse.patch（DCP-A，状态：**待 ⓪ 门放行**）
让 gfx90a 生产链（单发 prefill-ragged Triton 内核）产出每 (token,head) 的
LSE（自然对数底；空行 = -inf；HAS_ATTN_SINK 分支同样折叠），后端 _forward_mla
Triton 分支返回真实 lse（缓冲复用，cudagraph 安全）。
不传 lse 时 WRITE_LSE=False，内核行为与现状**逐比特一致**（默认路径零风险）。
AITER opus / 非 ragged 分支被要求 LSE 时显式 NotImplementedError（不许静默 None）。

生成/重生成：python3 make_patch.py（锚点不唯一会硬失败；树被并行会话改动后需重生成）。
应用：cd /home/qiba/ai/recipes/patches/vllm/vllm-openai-rocm-nightly-0918/core/tree && patch -p1 < ...0001...
（先 --dry-run）；回滚 patch -p1 -R。

### 上树后的尺子（按序，全部在空闲卡窗口跑）
1. 回归：WRITE_LSE 默认关 ⇒ 起服输出与关补丁前逐 token 一致（针尖复跑即可）。
2. 单测（小尺寸 GPU）：_sparse_attn_prefill_ragged_kernel 的 lse vs torch 参考
   masked logsumexp(qk*scale) —— 相对误差 1e-3。
3. 合并等价（2 卡即可）：上下文对半切两 rank ⇒ (out_A,lse_A),(out_B,lse_B)
   的 LSE 加权合并 == 全量单发 (out,lse)。fp32 累加，公式同 reduce 内核。
4. 才进 ① DCP=2 起服 + 128K 针尖。

### 已知设计约束（做 C 之前必读）
- **sink 双计陷阱**：本补丁把 sink 折叠进每行 LSE。DCP>1 时每 rank 的 partial
  各折一次 sink ⇒ 全局 softmax 分母被多算。DCP 合并路径要求：分片内核
  attn_sink=None，merge 完再折 sink（后端 AITER 分支的 torch.logaddexp 折法可抄）。
  GLM-5.3 是否有 sink 要在 ① 起服时打印 self.sinks 确认。
- 1M 阶段性能后手：单发内核 decode 行循环 ∝ 上下文/DCP；split-KV 路径
  （partial+reduce，LSE 素材现成）是 TPS 备选杠杆，另立 0002。
- 并行 OOM 修复（fp8_mqa_logits_torch 按 M 分块，17:38 由并行会话落码、
  本会话审查通过）与本补丁正交；长 N 下 m_chunk 退化为个位数的启动开销
  问题留给 0003（融合 logits+top-k Triton）。

## 0002_gfx90a_indexer_dcp_topk.patch（DCP-B 的 indexer 侧，状态：待 ⓪ 门放行）
上游 `_merge_dcp_topk_global`（各 rank 本地 top-K → 全局 top-K）是 **CuteDSL-only**，
gfx90a 上起 DCP 必抛 "DCP sparse-indexer merge requires CuteDSL"。
实测拆解：该路径里 **pack 本身就是 Triton**（`dcp_indexer_cutedsl.PackDCPTopkCandidatesKernel`），
只有最后的 stable-topk 选择器是 CuteDSL ⇒ 本补丁把那个 Triton pack **原样搬进**已挂载的
`model_executor/layers/sparse_attn_indexer.py`（不用改 launcher 挂载表），并用 torch.topk
复现 stable-topk 语义（-inf → -1；并列次序差异已在代码注释与笔记里写明）。
CUDA 侧仍走 CuteDSL 原路，行为不变。dry-run 干净、py_compile 通过。

### 起 DCP 的完整"必须翻"清单（2026-09-20 读码穷举）
| # | 位置 | 内容 | 状态 |
|---|---|---|---|
| 1 | attention impl `supports_dcp`(rocm_aiter_mla_sparse.py:775) | `False → True` | 待写（0003） |
| 2 | indexer `_assert_cutedsl_dcp_merge_supported` | CuteDSL-only 门 ⇒ 换 gfx90a 分支 | **0002 ✓** |
| 3 | attention builder `build()` | `triton_convert_req_index_to_global_index` → 上游 `triton_filter_and_convert_dcp_index`（`return_valid_counts=True, compact_valid_to_front=True`），`sparse_seqlen` 用其 valid counts | 待写（0003） |
| 4 | attention impl **能力声明** | `can_return_lse_for_decode = True` + `lse_base_on_e = True`（ln 底）——**合并由层做**（`mla_attention.py` 的 `dcp_manager.combine`），后端只返回 `(out, lse)` | **0003 ✓（首跑实测修正）** |
| 5 | 配置钉死 | `dcp_comm_backend`（上游只验证 `ag_rs`）、`cp_kv_cache_interleave_size`；`max_dcp_size=TP//kv_heads=8` ✓；`reorder_batch_threshold` 已被后端强制为 1 ✓ | 起服时核对 |
| 6 | 起服前裸测 | `--max-model-len 1048576`（不开 DCP）读 `Available KV cache memory`，提前判 ② 装不装得下 | 待跑（卡空闲） |

（不构成门控：candidate_blocks 的 `assert dcp_world_size == 1` 只走 v4.1 两级候选过滤，GLM 不用；
`_validate_dspark_dcp_support` 只在 spec-decode method=dspark 时生效。）

## 0003_gfx90a_attention_dcp.patch（DCP-C，**依赖 0001 先应用**，状态：**已上树，2026-09-20 首跑后修正**）
全在已挂载的 `v1/attention/backends/mla/rocm_aiter_mla_sparse.py`（3 hunks / 99 行）：
1. 能力声明三件套：`supports_dcp = True`、`can_return_lse_for_decode = True`、`lse_base_on_e = True`；
   `__init__` 取 `dcp_world_size/dcp_rank/cp_interleave`，并**硬拒 DCP+sinks 组合**。
   ⚠️ **不带合并**：首跑报 "DCP requires attention implementations to return the softmax LSE"
   后读码定案——合并由层 `mla_attention.py` 的 `dcp_manager.combine(attn_out, lse, ...)` 完成，
   后端自己再合并就是双重合并、静默算错（我最初的 `_merge_dcp_partial` 已撤销）。
2. `forward_mqa` 的索引转换改 dispatch：DCP>1 走上游
   `triton_filter_and_convert_dcp_index(compact_valid_to_front=True)` + 现成的
   `fetch_id_to_ragged_triton` 打成 ragged 扁缓冲；**`paged_kv_indptr` 契约不变**
   （行内超出部分本来就是 -1 空洞，稀疏内核掩 slot<0）。
3. `_merge_dcp_partial`：`all_gather(out)+all_gather(lse)` → fp32 ln-LSE 加权和
   （空行 -inf ⇒ 权重 0；全 -inf 行输出 0），数学与 ops 的 split-KV reduce 内核同构。

### 补丁栈验证（2026-09-20，scratch 副本，真树未动）
`0001 → 0002 → 0003` 顺序应用成功、三个文件 py_compile 通过、**反向回滚后与原文件逐字节相同**。
真树 mtime 保持 13:11 / 15:31 / 17:38（未触碰）。

### ① 的判据工具
`dcp_parity_probe.py`：同一 prompt（唯一句填充，避免 top-k 共振）分别对 DCP=1 / DCP=2 起服跑贪心，
比对 token 序列 + logprob，报首个分歧位置。**这是 ①③ 的正式判据**（针尖降级为 ④ 质量回归）。

## 0007/0008/0009 —— 队列对齐（2026-09-21，把 09-20 20:33 之后树上做的三组改动冻结入队）

生成方式（**机械**，不手改队列里的代码）：`python3 make_patch789.py` —— 先把 `base/*.orig`
铺到真实相对路径、按序打完 0001..0006 得到**重放中间态**，再对中间态与线上树做 diff，
按 hunk **内容嗅探**把 9 个 hunk 分给三片；归属不唯一/未知的 hunk **硬失败**（不许静默丢弃）。
生成后自证：全新重放 + 0007/0008/0009 必须与线上树**逐文件 sha256 相等**
（ops `168d0ba0c339` / backend `a68741014ac1` / indexer `8dc3afb0700a`）。
日常门禁：`python3 quark-int8/verify_patches.py` 的 ①。

### 0007_gfx90a_dcp_topk_ctx.patch（缺陷① parity；4 hunks：ops 3 + backend 1）
问题：请求路径上 `get_current_vllm_config()` 不可用（op 在 breakable-cudagraph 上下文里执行）
⇒ 实测 `AssertionError: Current vLLM config is not set`。做法：ops 定义模块级 `_DCP_TOPK_CTX`
+ `set_dcp_topk_ctx(...)`；attention 后端 `__init__` 里写入（那里 config 上下文可用），op 侧只读。
尺子：DCP=8 @32K/@256K 事实召回 6/6，日志不再出现该 AssertionError。每请求写一次，无性能影响。
默认：**总是生效**（DCP>1 必需）。

### 0008_gfx90a_dcp_debug_switch.patch（DCP 取证开关；backend 3 hunks）
`import os` + 两处 `MI250_DCP_DEBUG` 打印（`[DCPDBG out]` 的 out/lse 统计、`[DCPDBG idx]` 的
valid/lens/空洞统计，各限前 3 次）。默认**关**（env 空或 0 时零开销）。
尺子：设 `MI250_DCP_DEBUG=1` 能打印；不设时与关补丁逐 token 一致。

### 0009_gfx90a_sparse_splitk.patch（sparse split-K；ops 2 hunks）
一次 launch 把 (query, split) 当作行（`_splitk_make_indptr` 造 `[M*S+1]` indptr、查询优先行序
`r = i*S + s` ⇒ 源 ragged_indices 无需重排），再用 `_splitk_merge` 做 LSE 合并。
开关：`MI250_SPARSE_SPLITK=8`（默认 **0=关**）、`MI250_SPARSE_SPLITK_MAXM=8`。
尺子：内核与 S=1 对拍 bf16 1 ulp；事实召回开/关都要 6/6。
⚠️ **端到端为负**：单流 −12%（内核 7.2x 但 NCCL 141→255 µs、elementwise/copy 调用
1332→3439/step）⇒ 默认必须保持 0。详见 `hyperloom/reports/models/glm53-int4/decode-splitk-attention.md`。
⚠️ 低层函数是**返回** out（内部 `empty_like(q)`），不能读预分配缓冲——踩过，表现为 out 全 0。

### 队列的覆盖边界（已知盲区，别当成已覆盖）
`verify_patches.py ①` 只覆盖队列记账的 3 个文件。树上另有 5 个文件在 09-20 被改过但
**没有 base 原件 ⇒ 本门无法校验**：`model_executor/model_loader/weight_utils.py`（19:40，FST
装载速度，见 DCP_A_NOTES「装载速度」节）与 `models/deepseek_v41/{amd/vl_model.py, common/engram.py,
common/engram_fp8.py, compressor.py}`（06:17–07:30）。补法二选一：补 `base/*.orig` + 补丁，
或在 README 明确声明这些文件不属于本队列。

### 关于「6 个补丁全 rc=2」的更正（2026-09-21）
早前记录把该现象归因于「多文件 hunk / 目标文件没就位」——**归因错了**。真因是
`patch -i <相对路径>` 的路径按 **cwd（临时树）** 解析 ⇒ 补丁文件本身找不到 ⇒ rc=2，与补丁内容无关。
修法一行：`-i str(p.resolve())`。更正后 0001..0009 全部 rc=0，队列与树逐字节一致。

## 现场
- ⓪ 复跑：quark-int8/logs/glm53_verify32k_0920_1738.log（job bash-89，含
  16K 针尖 HIT/MISS 判定；17:38 的重启顶掉了并行会话 2 秒前的容器，
  已在 logs/.verify32k.lock/info 留致歉与交接说明）。
