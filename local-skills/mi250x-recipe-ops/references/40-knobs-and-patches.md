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

