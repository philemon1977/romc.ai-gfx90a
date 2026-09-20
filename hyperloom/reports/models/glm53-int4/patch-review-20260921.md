# 补丁全量评审（2026-09-21 06:1x，纯 CPU 取证）

## 一、清单与状态（"在树"= 线上挂载的补丁树；"在队列"= `quark-int8/dcp_patches/*.patch`）

| # | 补丁 | 落点 | 在树 | 在队列 | 启用 | 验证证据 |
|---|---|---|---|---|---|---|
| 1 | 稀疏分支写 LSE（0001） | `v1/attention/ops/rocm_aiter_mla_sparse.py` | ✓ | ✓ 0001 | **开**（DCP 必需） | DCP=8 @32K/@256K 事实召回 6/6 |
| 2 | indexer 局部→全局 top-k 合并（0002/0005） | `.../sparse_attn_indexer.py` + ops | ✓ | ✓ | **开** | 同上（修掉乱码） |
| 3 | attention 后端 DCP（0003/0004/0006，含逐行本地长度） | `v1/attention/backends/mla/…` | ✓ | ✓ | **开** | 同上（修掉负长度） |
| 4 | **模块级 `_DCP_TOPK_CTX`** | ops | ✓（7 处） | **✗ 未进队列** | **开** | 修掉 `AssertionError: Current vLLM config is not set` |
| 5 | MoE GEMV v3（scale 提出循环） | `moe_gemv/mi250_moe_gemv_gs.py`(+v3) | ✓ | n/a（独立模块） | **开**（默认 v3） | 微基准 gemm1 8.7x；端到端单流 +45%；召回 6/6 |
| 6 | worker 内 profiler | `moe_gemv/sitecustomize.py` | ✓ | n/a | **关**（`MI250_PROF_WORKER`） | 用它拿到 decode 归属表 |
| 7 | **decode 稀疏注意力 split-K** | ops | ✓（8 处） | **✗ 未进队列** | **关**（`MI250_SPARSE_SPLITK=0`） | 内核 7.2x、对拍 bf16 1 ulp；**端到端 A/B 为负 −8~−19%** |
| 8 | **QuickReduce C2+C3**（移植前人） | `quick_all_reduce.py` + `cuda_communicator.py`（**仅容器 FS**） | n/a | ✗（applier 在仓库） | **未启用**（三条 env 未设） | 待第 2 步：日志出现 QUICK_REDUCE + 召回 6/6 + TPS |
| 9 | Hyperloom MI250X 身份/runner | `hyperloom/patches-local/*.py` + magpie 脚本 | ✓ 已应用 | applier 在仓库 | 已生效 | `gpu_type=mi250x`、runner 折叠、HW_SPECS 命中（比值 0.309） |

## 二、三个必须修的问题（按严重度）

### 1. `dcp_patches/*.patch` 已落后于树两处（会误导复现）

实测：队列里 `_DCP_TOPK_CTX` 出现 **0** 次、`split-K` 出现 **0** 次，而树里分别是 7 / 8 次。
⇒ 任何人按 `dcp_patches/README.md` 打补丁，得到的是**会崩的旧版**（缺少 DCP_TOPK_CTX 修复）+ **没有 split-K 代码**
（后者虽默认关，但"env 存在而代码不存在"更容易误导）。
**修法**：把这两处写回 `make_patch*.py` 并重跑生成，或补 `0007_patch_tree_parity.patch`；并加自检断言。

### 2. QR C2/C3 只活在容器文件系统里（不可追溯 + 会丢）

容器重建即消失，且"镜像态 ≠ 运行态"。**修法二选一**：写进 `Dockerfile.glm53-hyperloom` 重建镜像；
或在起服脚本里**先跑幂等 applier**（`apply_gfx90a_quickreduce.py`）。

### 3. applier 类补丁缺"功能断言"，已两次出现"打了但静默无效"

- 今天踩到：QR applier 的 `QR_NEW` **漏逗号** ⇒ Python 隐式拼接 ⇒ `supported_archs` 赋值行**整行变注释** ⇒
  运行时 NameError 被 `try/except` 吞掉 ⇒ **QR 看起来"补丁打了却仍不启用"**（`ast.parse` 查不出来）；
- 更早踩到：split-K 分支去读预分配缓冲而不是**捕获低层函数返回值** ⇒ out 全 0（lse 却正确）。
**修法**：每个 applier 除 `ast.parse` 外必须带**功能断言**（QR：断言赋值行未注释 + marker 存在；
split-K：断言 `out_s = _rocm_sparse_attn_prefill_ragged_triton(` 被捕获）。

## 三、其它观察（低风险但记录）

- **GEMV 三处副本当前哈希一致**（tree / `quark-int8/moe_gemv_patch/` / 容器）：
  `af079db138ab` / `07b0d78f40b0` / `31e8573f5500`（三个文件），但**没有自动检查** ⇒ 建议纳入自检脚本。
- 判定"今天改了什么"**不要用 mtime**：树里 4 个文件的 mtime 混着早前会话的改动；应用 grep marker。
- 树里残留早前会话的历史文件（`*.bak_pre_oom_fix_0920`、`*.orig` 等）：不影响运行，但让"哪份是真值"含糊，
  建议清理或在 README 标明真值来源。

## 四、建议的收尾动作（都不需要 GPU）

| # | 动作 | 产出 |
|---|---|---|
| a | 对齐补丁队列 + 写 `quark-int8/verify_patches.py`：断言 ①tree == base+patches（逐文件）②GEMV 三处哈希一致 ③QR marker/赋值行正确 ④split-K 默认关 | 一条命令给出"补丁是否自洽" |
| b | QR 落地方式定案（进镜像 / 起服前 applier） | 消除镜像-运行态漂移 |
| c | 清理历史冗余文件并更新 `dcp_patches/README.md` 的真值说明 | 复现路径唯一 |
