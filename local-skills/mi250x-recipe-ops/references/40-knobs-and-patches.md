# 跨臂开关与补丁

> 判「是否适用」用 `data/*.json` 的 `applies_to`（九轴谓词）；判「有收益吗」用 `effect_by_scope`（分档效果）。两者是两件事。

<!-- ── 搬运自 SKILL.md L403-473 ── -->
> **T1/T2 混杂** — 自研 kernel 与稀疏 split-K 开关（T2 量化相关）、roofline 口径（T0 方法）、KB 回填（技能自身）。

## 本会话新增（2026-09-21 第三轮：MoE GEMV kernel、稀疏 split-K 开关、roofline 口径、KB 回填）

来源：GLM-5.3-CT-Int4-W4A16 / TP8+DCP8 / gfx90a 的实测与取证，报告在
`hyperloom/reports/models/glm53-int4/`。以下四条此前在本技能里 **0 命中**。

### MoE 专家 GEMV（自研 kernel，目前唯一已收回的 kernel 级杠杆）
- 问题：decode 是 M=1 的 GEMV，专家权重每个 token 全读一遍；把 **scale 提到 k 循环外**是
  这一步的关键（v3）。
- 开关（launcher 已接好，`PYTHONPATH=/patches/moe_gemv` 由 launcher 注入）：
  `MI250_MOE_GEMV=1`（默认开）、`MI250_MOE_GEMV_MODULE=mi250_moe_gemv_gs`、
  `MI250_MOE_GEMV_KERNEL`（v3 = scale hoisted）、`MI250_MOE_GEMV_BOTH`、`MI250_MOE_GEMV_DEBUG`。
- 实测：gemm1 **8.7×** / gemm2 **5.0×**（生产分片形状）；端到端单流 6.38–6.81 → **9.60–10.71 tok/s**；
  并发 32 聚合 33.8 → **59.1 tok/s**；事实召回仍 6/6。
- **v2 是被证伪的那一版**（单流 6.8 / 聚合 33.8 ≈ 等于不开），只留作对照；别把 `_v2` 当可用模块。
- 三处副本必须一致（补丁树 / `quark-int8/moe_gemv_patch/` / 容器），门是 `verify_patches.py ②`：
  `gs=af079db138ab` / `v2=c96264af84c1` / `v3=07b0d78f40b0` / `sitecustomize=31e8573f5500`。
- ✂️ **这条杠杆已经收回**：decode 归属表里 MoE GEMV 只占 **1.8%** 的步时间，别再去这里找收益。
  报告：`moe-gemv-scale-hoist.md`。

### 稀疏注意力 split-K（`MI250_SPARSE_SPLITK`）——与 0.28 的 split-KV **不是一回事**
- 本技能别处的「split-KV」指 `VLLM_ROCM_SPLITKV_PA`（paged-attention 的 KV 切分）；这里是
  **gfx90a 稀疏注意力路径自己的 split-K**：一条 launch 把 `(query, split)` 当行
  （`_splitk_make_indptr` 造 `[M*S+1]` indptr、查询优先行序 `r = i*S + s` ⇒ 源 ragged_indices
  无需重排），再用 `_splitk_merge` 做 LSE 合并。补丁：`quark-int8/dcp_patches/0009_gfx90a_sparse_splitk.patch`
  （队列可重放：`base + 0001..0009 == 线上树`，逐文件 sha256 相等）。
- 开关：`MI250_SPARSE_SPLITK=8`（**默认 0=关**）、`MI250_SPARSE_SPLITK_MAXM=8`（只在 M≤该值生效）。
- 结果：内核 7.2×（M=1）/ 2.4×（M=4），与 S=1 逐位一致（bf16 1 ulp）；但**端到端单流 −12%**
  （NCCL 141→255 µs、elementwise/copy 调用 1332→3439/step）⇒ 默认必须保持 0。
- ⚠️ 静默错陷阱：低层 `_rocm_sparse_attn_prefill_ragged_triton` 是**返回** out（内部 `empty_like(q)`），
  不能读预分配缓冲 —— 否则 out 全 0 而 lse 正常，看起来「没崩」但结果全错。

### Roofline 口径：本机的慢**不是带宽**问题（别再按带宽解释）
- `T_mem(mi250x) @num_gpus=8 isl=osl=1024 conc=32 = 859.0 tok/s`（BW 13.1 TB/s）；同参数 mi300x
  = 2779.3 ⇒ **比值 0.309 是防「跑在 MI300X 口径下」的假通过判据**。
- 实测对照（同一把尺子）：单流 @ctx≈800 = 10.2 tok/s（**1.19%**）；conc8 41.4（4.82%）；
  conc32 59.5（**6.93%**）。
- 原因：decode 一步 186.7 ms 里 sparse-attn 67.2（36%）、NCCL 36.3（19.4%）、
  `triton_w4a16_gemm` 35.3（18.9%）、我们的 MoE GEMV 3.3（1.8%）；权重流量只有
  **18.6 GB/s/rank = 峰值的 1.1%** ⇒ 受限在**层内串行/启动延迟**。
- 正确表述：「每步的层内串行开销吃掉 93% 的访存预算」，而不是「带宽不够」。
  报告：`decode-step-attribution.md`（含 roofline 一节）。

### RecipeKB 回填（GLM-5.3 int4 / mi250x）与三个静默坑
- 入口：`python3 scripts/note_glm53_int4_kb.py`（`--dry-run` 可预演；用 Hyperloom 自己的
  `LocalRecipeStore.put_recipe`，别手写 recipe.json，否则 `history/vN` 与 `version` 脱节）。
- 三个**静默**失败（今天全踩过）：① `remaining_gaps` 条目必须是 **dict**（`description`/`metrics`），
  写字符串会被直接丢弃；② `kernel_optimizations` 是**定长 dataclass**，键名不对会得到一串**全零**条目；
  ③ `best_config.extra_envs` 会被 warm-replay **当环境变量注入** ⇒ 别往里写说明文字。
- 权限坑：Hyperloom 容器以 root 写的槽位是 `root:root 0600`，宿主读不到 ⇒ `local_store.search()` 抛
  `LocalRecipeStoreError`、`verify_recipe_kb.py` 失败；修法：容器内 `chown -R 1000:1000` + `chmod 644`。
- 两条 warm-start 语义：`_DEFAULT_WARM_REPLAY_MIN_CONFIDENCE = 0.7`（低于它只被读、不会复现其
  `best_config`）；recipe 的 `what_failed` 会被注入 explore 的 rejected 账本 ⇒ **负结论写进去等于
  省下一次重测**。
- 磁盘行数可以多于 `seed_manifest.json`（实测 18 vs 12）：本库有**三种写入者**（播种 / Hyperloom 自己 /
  会话回填），见 `hyperloom/kb/README.md`。

### 待办（别当成已解决）
- QR（QuickReduce）在 **GLM-5.3 int4 上仍未验证**：A 臂基线已测（QR 关：召回 6/6；
  ISL≈800/OSL128 请求级 conc 1/8/32 = 4.38 / 28.15 / 74.77 tok/s —— 与 1024/1024 档的 59.1
  **不同尺子，不可互比**）；B 臂（QR **真开**）**至今没出过任何结果**。
- 进展（2026-09-21 10:40）：入口 bug 已被重烤修好（`docker inspect` 与 stock 镜像 Config 逐字一致，
  实跑 `vllm serve --help` 正常）。**A2 臂**（同一个 `-qr` 镜像、三条 env 全不设）已测：召回 6/6、
  请求级 TPS 4.18 / 26.64 / 73.13；对照 **A 臂**（stock 镜像）4.38 / 28.15 / 74.77 ⇒ 差 **−2…−5%**，
  落在本机 cross-boot 漂移内 ⇒ **仅凭单次对拍不能说"代码在但关着是中性"**，只能说没有反向证据。
  两臂日志里 `tp:0`/`ep:0` 都选 `['PYNCCL']`。
- 现在要跑 B 臂：`SKIP_A=1 SKIP_A2=1 bash quark-int8/qr_ab_watch.sh`（镜像 `...-0918-qr` 已可用；
  脚本里的 `wait_free` 会等到八张卡都空才动手）。


---



---

## 附录 A · 注意力内核的真正锁在哪（**T1/T2**）

> 出处 `docs/MI250X-attention-hd256-内核缺口-2026-09-07.md`、
> `docs/MI250X-hd256-customPA-静态审计-2026-09-07.md`、
> `docs/MI250X-GLM53-准入审计-2026-09-07.md`。
> **这一节的存在理由是防止重做**：三条都推翻了看起来合理的判断。

- 🔑 **真正的锁是 `block_size`，不是 `head_dim`**。
  `rocm_attn.py:179-183` 源码原文：native C++ kernel 因 shared memory (LDS) 约束
  **只支持 block size 16 和 32**；vLLM 允许任意 16 的倍数，但那是 **Triton 路径**。
  而混合模型的 block 由 **`_align_hybrid_block_size()`（`platforms/interface.py:778`）**决定
  （要保证注意力页 ≥ mamba/GDN 状态页）⇒ 实测被抬到 **400/528** ⇒ 幂次门失败。〔§9.2 L270-282〕
- 🔑 **`hidden_size // num_attention_heads` 推导 head_dim 对 MLA/稀疏模型无效**。
  GLM 是 MLA（`kv_lora_rank=512`、`qk_nope_head_dim=256`），且 `layer_types`
  = 11×`deepseek_sparse_attention` + 34×`linear_attention`、**没有任何 full_attention**
  （`index_topk=2048` ⇒ 每步只读 2048 token）。本机那份扫描表里 GLM 的 head_dim 就是这么错的
  ——`HEAD_SIZE=64` 本身是 bug 产物（config 里 `head_dim: 0` 未声明），真实 qk 维 **256**。〔§9.1 L258-268；准入审计 §0 L7-8〕
- ❌ **加 `case 256` 已实测 NO-GO 并回滚**：microbench **488.3 GB/s** vs Triton 参考 12.3
  （**快 39×**）但**算错**。⇒ 别再重做。附带判据见 `references/60-…` §3（不能 `grep 256`）。〔§7 L114-138〕
- 🛑 **不要再把 GLM 当 custom PA 的靶子**：`use_rocm_custom_paged_attention` 全树**只有一个调用者**
  （`chunked_prefill_paged_decode.py`，稠密分页 decode），而 GLM 的注意力是 **MLA+DSA**，
  走 `ROCMAiterMLASparseBackend`。要继续就改进 **Triton 参考内核 `kernel_paged_attention_2d`**
  （它在 hd 64/128/256 上**一律只有 7~12 GB/s**，受控对照已证慢与 head_dim 无关）。〔准入审计 §5 L138-151〕
- **单 workgroup 的坐标**（解释长上下文墙）：Triton 参考内核 `grid = (num_seqs, num_kv_heads)`、
  `chunked_prefill_paged_decode.py:152 for j in range(0, num_blocks)` **单 program 串行扫完整条上下文**；
  Ornith TP4 每 rank 1 个 kv 头 ⇒ 单流只有 **1 个 workgroup = 104 CU 的 1.0%**。〔§10 L315-336〕
- ⚠️ **别把这条外推到 8107**：8107 的 QSA 每步只读 2048 token ⇒ attention 占比恒 ~0.1%。〔§6 L102-113〕
- **若真要动 `attention.cu`**（唯一施工坐标）：`shared_logits[NWARPS][4][16][4]` 的 **dim0 被当成
  `NWARPS` 用**（写侧 `offset1 = lane16id/4 ∈ [0,4)`），却被 **`QKHELOOP`** 索引
  （hd256 时 `QKHELOOP=8`）⇒ **dim0 越界**；根因是 Q 的共享内存装填把 head 维**钉死在 128 个元素**
  （`lane16id × CONTIGUOUS_SCALAR_ELEMS_16B = 16×8`）。派生常量：`QKHELOOP = HEAD_SIZE/32`
  （64:2 / 128:4 / 256:8）、`VHELOOP = HEAD_SIZE/64`；寄存器驻留从 hd128 的 64+64 涨到 hd256 的
  **128+128 VGPR**。⚠️ **溢出与 40× 慢是同一个二进制测出来的 ⇒ 别把溢出当根因**。
  复现必须**先 hipify**（否则编的是旧 `.hip`）；`L1623 __GFX11__ / L2382 __GFX12__ / L3126 #else`
  **与本机无关，不要读**。〔customPA 静态审计 §1-§6〕

## 附录 B · llama.cpp 侧的两个"看着像性能问题、其实是路径切换"

> 出处 `docs/MI250X-GLM-5.3-Flash-本机运行全记录-2026-09-05.md`。**适用域：GLM-5.3-Flash / Q8_0 /
> llama.cpp `949f7efb0`(b136) 与 `629b50552`(b223) / ROCm 7.2.4 / 8108。**

- ★ **「9 行悬崖」源码判据**：`mmvq.cuh:3 #define MMVQ_MAX_BATCH_SIZE 8` ⇒
  Q8_0 `ne11 ≤ 8` 用 **MMVQ**、`≥9` 切 **MMQ**；k-quant **4 行即切**；
  **MoE 专家路同一阈值 ⇒ dense 与专家路同时翻面**。〔§6.6 L371-382〕
- ★ **「跨悬崖」是陷阱**：`-np` 8→9 聚合 **58.75 → 44.12（−25%）**；
  运行时证实 MMVQ 49.2%→1.3%、MMQ 0→67.3% ⇒
  **正确目标是留在 ≤8 行并把 MMVQ 变便宜**，真吃 MMQ 需 ~128 行。〔§6.6 L409-432〕
- ⚠️ **上面这条"9 行悬崖"自带两条适用域限定**（原档 §6.11 L561-563 自己划的边界，
  离开这两条就不得引用）：
  ① 它是 **`np` 维**的结论（np=9 是 9 条独立序列各读一遍权重、摊薄不了），
     **不适用于投机那一维**；决定成本的是"每推进一个 token 要付几次整权重读"，
     不是"单次 verify 多塞几行的边际成本"；
  ② **只在 Q8_0（`ne11<=8`）上成立**——换 k-quant 门槛降到 3 行，
     **不能直接搬**，`UD-Q4_K_XL` 那轮需重测（要专设一档 `NGRAM_MAX=2`）。
  同轮另一条产出是**反向的**：压 `NGRAM_MAX` 64→16 没让代码档回升（25.91→25.38，噪声内），
  反而把复述档砍掉 40% ⇒ **agent 默认值维持 `NGRAM_MAX=64`**（"该不该调小"判成"不该"）。
  注意 `NGRAM_MIN` 与 `NGRAM_MAX` 是**两个旋钮**，agent 配置里同时是
  `NGRAM_MIN=16` + `NGRAM_MAX=64`（L573）——别把下面 `n_min` 那条读成"要调小 NGRAM_MAX"。
- **`ngram-mod` 默认 `n_min=48` 是「丢弃门槛」**（草稿短于 `n_min` 整个作废）
  ⇒ **不调到 ~16 等于没开**。
  ⚠️ 技能里其它 `ngram` 命中指的是 **engram**（DSV4.1 的表），**不是同一件事**。〔§6.7 L490-492〕
- **Q4 的价值是显存不是速度**：`UD-Q4_K_XL` 186.0 GiB = Q8_0 的 **0.586×**（字节 −41%）⇒
  速度 **−1.9%**（噪声内、方向为负）；@1M/4die = 229.1 GiB ⇒ **2 份 1M 实例首次可行**。
  且「总量 ÷ 卡数」的显存估法在 `-sm layer` 下**无效**。〔§6.9 L627-632, L655-663〕
- **达成率的分母必须是单 die**：`-sm layer` 下任一时刻只有约 1 张 die 在读 HBM ⇒
  正确上限是**单 die 实测 1247 GB/s**，不是 `8×1247`（原报的 3.5–4.8% 是单位错配、**低估 8 倍**）。〔§6.8 L590-591〕
- **Q8_0 没吃到任何整数算力，连"少读字节"的红利都没兑现**：整个 HIP/CUDA 后端里
  `v_mfma_i32` / `_i8>` 命中 **0 个文件**；`ggml_cuda_dp4a`（`common.cuh:704`）在 HIP+CDNA 下编成
  `__builtin_amdgcn_sdot4`，但调用方是 **k-quants / IQ 系**的解包-点积内核；
  **Q8_0 的实际路径**在 `ggml-cuda.cu:5236-5241` —— 与 `Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL`
  一起被列进 **dequantize 回退白名单** ⇒ 解量化成 float → fp16 MFMA / fp32 FFMA。
  〔`docs/下载-ModelScope-Ornith-1.5-397B-Q8_0-与8110底座-2026-09-09.md` §8 L159-167〕
- **Ornith 混合架构的 KV 便宜到离谱**：60 层里只有 15 层存 KV（`full_attention_interval=4`）、
  `kv_heads=2`、`head_dim=256`、f16 ⇒ **30 KiB/token**；45 层 GDN 只有递归态 ≈180 MiB/槽
  **完全不随长度涨**；MTP 草稿层再加 2 KiB/token。@1M 合计 32.0 GiB、最忙 die 59.0/64。
  纯算术天花板 ≈2.04M 但**不取**（368k prefill 期间占用比加载完再涨 +1.5 GiB）
  ⇒ **1M 是"留得住 5 GiB 运行时余量"的最大档**。〔同文 §13 L302-312〕
- ⚠️ **「这份 Q8_0 里没有 MTP 头」不代表模型没有**：源仓 2924 张量里 `mtp.*` 占 **1553**，
  且全部集中在单独一个文件 `model-mtp.safetensors` = 12.29 GiB ⇒ 是**转换器（bartowski）把 MTP 丢了**。〔§7 L134-139〕

## 附录 C · 上下文档位的真实约束是**乘积**

> 出处 `docs/上下文口径-单条会话最低256k-2026-09-13.md` §5.2 L74-84。**适用域：8112 / llama.cpp / DSV4.1。**

- 🔑 **约束是 `CTX × UBATCH` 的乘积**（约 ≤32 才守得住 ~14 GiB）：
  `256k/ub256` **根本没起来过**（die0 `sched_reserve` 17470 MiB）；
  `1M/ub256` 当场 OOM（申请 **68926 MiB**）；只有 **`1M/ub32`** ✅
  （die0 9292 → 最忙 ROCm6 14662 MiB、die0 驻留 60862/65520 MiB）。⇒ **ctx 与 ubatch 必须成对降。**
- **被推翻的判断**：`-fa on`（`flash_attn`）**不是**长上下文的前提——
  日志 `flash_attn = enabled` 确凿生效后 die0 仍申请 68.9 GiB。
  （那块显存是 **DSA 候选掩码**，在注意力算子**之前**进图。）
- **1M 装得下，代价是 prefill 掉到 1/3**（102 vs 303 t/s ⇒ 1M 冷 prefill ≈2.9 h）。〔§5 L95〕
- **每请求固定开销 ~0.5–0.9 s 与 `n_ctx` 无关**（CTX=1M vs 32k 差 ≤1.2%）
  ⇒「常驻 1M 每步罚钱」不成立，1M 的代价只有**首次 prefill**。〔GLM 全记录 §6.4 L936-948〕
- **slot 数由吞吐账否掉，不由显存账决定**：8112 账面能塞 3–4 条 256k，
  实测 `np=2` = 25.3 < `np=1` = 29.9 t/s。〔`docs/DeepSeek-launcher-上下文与slot账-2026-09-12.md` §3 L74-78〕
- ⚠️ **8103 的 KV 实账是 ≈6.6 KiB/tok，不是旧稿的 20 KB/tok**
  （SWA raw 32.25 MiB 固定项 + CSA 压缩 5376 + HCA 160 + LID 1344）；
  「1M 时驻留比权重大 ~20 GiB」的大头是 **compute buffer/graph（8 die ~14 GiB）不是 KV**
  ⇒ 按 20 GiB/slot 记账会把上限**算小 ~3×**。〔同文 §1 L20-32〕

## 附录 D · AITER 的 ASM attention 是「**有库、无调用方**」

> 出处 `docs/MI250X-AITER-射程-本机模型扫描-2026-09-07.md` §3 L78-96。

- vLLM decode 走**自己的** `torch.ops._rocm_C.paged_attention`；aiter 的 attention 后端
  引的全是 **Triton/Gluon**；`fmha_v3` 在全树**零引用**。
  ⇒ 接上需要**新写一个 attention backend**（**功能开发，不是打开开关，也不是重跑 repatch**）。
- 好消息：`module_attention_asm.so` 的 arch 码对象为 `[]`（运行时按 `hsa/<arch>/pa/` 装载 `.co`）
  ⇒ **移植产物放进去能被找到**。
- 9 模型逐字段射程扫描表（head_dim / heads / kv / GQA / 卡在哪一门）在同文 §2 L47-66 +
  `tools/aiter_admission.py` 逐门输出；最近的是 **Ornith-35B-A3B，只差 head_dim**。
- ⚠️ **该文 §4bis 的推论已被后续实测推翻**（形状与接线部分仍成立）：
  正确下一步是给 vLLM 加 `case 256` 重编，而不是 AITER/rocWMMA
  （后两者硬要求 `head_size ≤ 128`，对 hd256 模型**永远**不适用）。
  —— 而 `case 256` 本身也已判 **NO-GO**（见本文件附录 A）⇒ **这条路两头都关着**。
- ⚠️ **仍未证实、别当结论**：`block_size` 必须 2 的幂、KV 必须 native layout
  （`has_native_layout`）否则强制 Triton。

## 附录 E · ATOM 插件：墙 1 的机理与那条必需 env

> 出处 `docs/MI250X-ATOM-vLLM插件-本机实测-2026-09-06.md`。**ATOM 在 gfx90a 已判死（四道墙）**，
> 但这两条机理可迁移到任何"bf16 模型为什么需要 fp8 指令"的疑问上。

- 🔑 **与权重精度无关，是编译单元的连带依赖**：`custom_all_reduce.cu` 的派发宏把
  「普通版」和「fp8 逐 token 量化版」写在**同一个宏的两个分支**里，再对 fp32/fp16/bf16
  各展开一次 ⇒ gfx90a 缺 `fp8-conversion-insts` 直接**编译失败**。
  （同理见 `references/60-…` §9：警告消失 ≠ 行为改变；这里是"宏在，就必须能编"。）
- 🛑 **`VLLM_PLUGINS=` 是 ATOM 场景的必需项**：ATOM 注册的是 vLLM **platform** plugin，
  会整体替换 `RocmPlatform` 从而替换 backend 选择；作者自己的全部实测配方都设 `VLLM_PLUGINS=`
  让 dispatch 回到 vLLM 自己的。DSpark 文档 §0 再次确认这是**起服必备**。
- 用 fork 逐墙对照证明「社区 aiter gfx90a 移植**救不了 ATOM**」，墙 1 只有**未 merge 的 #4389** 能治；
  且该 fork 自己的结论是 `AITER's RMSNorm in particular is unvalidated on gfx90a`，
  其配方**全部** `VLLM_ROCM_USE_AITER_RMSNORM=0`。
- **QR 的 `MIN_SIZE` 覆盖结论与上游自测相反**（我们用 `=0` 强行覆盖到 4–10 KB）；
  且「先看单流 decode，不要只看并发就下结论」。

## 附录 F · `mamba_cache_mode` 三档语义与 `all` 的内存几何

> 出处 `docs/Ornith-397B-Mamba-All-模式实现方案-2026-09-18.md` §0/§1bis。
> **适用域：混合 GDN/Mamba 模型（Ornith / Flash-Next）× vLLM。**

| mode | 语义（`config/cache.py:141-148`） | 实测代价 |
|---|---|---|
| `none` | 关前缀缓存 | — |
| `align` | **只在「某个 scheduler step 的最后一个 token、且该位置是 `block_size` 整数倍」时**才留状态快照 | **≈0**（107.48 vs 107.80 t/s，n=3 spread 0.0%） |
| `all` | 每个 block 边界都留 | **起服即失败**，见下 |

- ⚠️ `align` 的**推论**要写清：因为 decode 步的块尾通常不对齐 ⇒
  **上一轮生成的那段没有快照 ⇒ 多轮里这段要重算**。省了内存但没省多轮重算。
- 🔑 **`all` 的第一个拦路虎是内存几何，不是代码**：`all` 语义 = 每 block 边界留一份 mamba 状态
  ⇒ 单请求状态内存 **∝ `max_model_len / mamba_block_size`**（262144/544 ≈ **481 块**）⇒
  起服直接 `ValueError: … 16.47 GiB KV cache is needed, which is larger than the available
  KV cache memory (5.95 GiB)`。**这是"稠密快照"的固有代价，不是 bug。**

## 附录 G · 那条 9 行警告**为什么会出现**（机制，解释更正 3）

> 出处 `docs/Qwen4Exp-MTP前缀复用-判定-2026-09-18.md` §2 L24-32。

vLLM 有两条规则能让 MTP 草稿组被正确标注，**qwen4exp 两条都不中**：
1. `non_causal_multi_token_decode` 标记 —— **只有 MLA 会带**；
2. `use_deepseek_v4_fallback` —— 门被 `_is_deepseek_v4_eagle()` **限死成 `model_type == "deepseek_v4"`**。

而本模型的 MTP 草稿层是 `layer_type="full_attention"` 且
`mtp_start_layer_idx = num_hidden_layers`（`models/qwen4_exp/amd/mtp.py:171`）
⇒ **恰好是最后注册的层，形状与规则 2 完全一致，只是模型名不匹配**
⇒ 兜底分支 `_warn_if_unannotated_eagle_mamba()` 把**所有**组当草稿组并打出 9 行警告。

🔑 **这解释了为什么"补丁只消警告、不改行为"**：警告本身就是**按模型名误判**的产物，
不是"复用真的被禁用"的陈述。修法是 **env 门控、默认关**（不改变现役行为）。
⇒ 通用判语：**按名字设的门，换个模型名就可能误开/误关；先读门条件的输入是什么。**
