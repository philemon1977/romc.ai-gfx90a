# T1 · 版本层（引擎 / ROCm / PyTorch / aiter）

> 本文件每条都绑定具体版本。换版本前先读 `data/scope.json` 的 `known_confusions`。

<!-- ── 搬运自 SKILL.md L281-319 ── -->
> **T1 · 版本层** — 绑定 aiter 0.1.19 + torch 2.12.0+git6bbd260 + ROCm 7.2.4 这一组指纹。换任一轴，缓存**不可复用**。

## aiter JIT prebuilt reuse (never pay the ~50 min compile twice)

aiter's JIT does **not** reuse across envs: a fresh env compiles 72 CK instances for
`module_gemm_a8w8` (~50 min of CPU, no GPU needed) on first import. Reuse *is* natively
supported — the switch is `AITER_JIT_DIR` — and this host has a verified cache.

```bash
cd ~/.cache/aiter-gfx90a
python3 restore.py --list                                       # 426 MB cache, 2 variants
python3 restore.py --env <env> --variant full-72inst-production --apply
HIP_VISIBLE_DEVICES=<idle die> python3 verify_gemm.py            # required acceptance
```

- Mechanism (read from aiter 0.1.19 source): `compile_ops` does `get_module(md)` and only
  falls into `build_module(...)` when that raises `ModuleNotFoundError`; with `AITER_JIT_DIR`
  set, `get_module_custom_op` imports the module from that dir (it is put on `sys.path[0]`).
  The `.so` must be the **bare name** `<md>.so` — `JIT_EXTENSION_VERSIONER` is per-process
  memory state whose first `bump_version_if_changed` returns 0, so there is no `_v1` suffix.
  `_needs_arch_rebuild()` scans the `.so` for `amdhsa--gfx*` and passes it when gfx90a is
  present. **Trap:** `build_module`'s `MainFunc` starts with
  `os.remove(get_user_jit_dir()/<md>.so)` — entering the compile path deletes a working
  prebuilt. "Compile finished but no `.so`" is that delete plus a re-compile, not lost output.
- **Three gates, all required:** aiter version + torch version + ROCm version must match the
  archive, and the running arch must be in the `.so` markers. Measured here: three envs share
  aiter 0.1.19 (8316 sources byte-identical), torch §2.12.0+git6bbd260`, ROCm 7.2.4.
- Verified install (`vllm_master_rocm724`, 2026-09-21): `get_module` **0.300 s** (not 50 min),
  `[256,4096,4096]` int8 GEMM rel_err **6.369e-03**, `[1024,2048,2048]` **7.042e-03**
  (tol 2e-2). Evidence line to look for in any arm's log:
  `[aiter] import [module_gemm_a8w8] under …/aiter/jit/module_gemm_a8w8.so`.
- Reading the numbers: a `max_abs` of 0.5 is **one bf16 ULP at magnitude ~256**, not an error —
  judge on **relative** error. And `not found tuned config in a8w8_tuned_gemm.csv, will use
  default config` means correctness is still proven but **performance is not** the production
  config; top up the tuning table before timing anything.
- **Do not** assume same-named `.o` files are interchangeable across envs: measured
  **38/38 same-name `.o` hashes differ** between `wu1w-int8-028` and `vllm_0.28.0_rocm72`,
  and the cause is a build-target difference (`"gfx90a": 104` in `GFX_CU_NUM_MAP`), not the
  ROCm version (both are 7.2.4). Copy `.so`, treat `.o` as a fallback only.
- Full write-up: `hyperloom/reports/aiter-jit-prebuilt-reuse.md`.


<!-- ── 搬运自 SKILL.md L340-361 ── -->
> **T1 · 版本层** — 与 AMD 官方技能 `serving-llms-on-instinct` 的仲裁表。官方技能按 gfx 架构分档且只覆盖 gfx942/gfx950。

## 与 serving-llms-on-instinct（AMD 官方技能）的仲裁（2026-09-21）

该技能直接读 `data/gpu_overrides.json > gpu_configs`（按 **gfx 架构**分档），而它只覆盖
`gfx942`/`gfx950`（MI300X/325X/350X/355X）—— 全技能内 MI250/gfx90a 命中 **0** 次。
所以落在本机的任务，它给的是通用/MI300 口径。**本机任务一律以本技能为准**，冲突处按下表：

| 通用口径可能怎么说 | 本机实测事实 |
|---|---|
| 启用 AITER 加速 | **不可用**（gfx90a 无 AITER MoE 路径） |
| 打开 QuickReduce | **禁用**：`init_custom_qr` 固定吃 ~9 GiB/卡 ≈ 本模型 KV 全部预算 |
| 用 split-KV 提速注意力 | **默认 0**：conc 1/8/32 三档全负（−18%/−14%/−13%） |
| 关掉 eager 换 CUDA graph | 需同时给 `MAX_CUDAGRAPH_CAPTURE_SIZE>=1`，否则断言拒绝 |
| indexer 走框架默认 | **必须 `DSV41_IDX_AITER_KERNEL=1`**：默认是更慢且本机不可信的回退，开启 +69.3%（conc32） |

- 桥接（幂等，bootstrap 还原第三方文件后要重跑）：
  `python3 hyperloom/patches-local/apply_serving_skill_mi250x.py`（`--check` 只看状态，`--revert` 回滚）。
- **SKU 级而非架构级**：64 GiB/GCD、104 CU/GCD、TP8 时权重 52.9 GiB/rank ⇒ 32k 档 KV 仅
  8.17 GiB / 94,016 tokens。别按「MI250X = 128 GiB」算 KV（那是两个 GCD 之和）。
  （2026-09-21 补正：52.9 GiB/rank 与 5.89 GiB KV 同属 **DCP=8** 那一行；**DCP=1** 是 **50.84 GiB/rank + 8.17 GiB**。
  出处 `hyperloom/reports/models/glm53-int4/decode-config-ablation.md` §四。）



<!-- ── 搬运自 SKILL.md L362-402 ── -->
> **T1 · 版本层 / T0 混杂** — DCP 与 fp8 KV 起法、补丁队列自检、Docker 自建镜像的坑、数字口径。（下一步应再细分：Docker/口径属 T0，DCP/补丁队列属 T1。）

## 本机运维增补（2026-09-21 第二轮：DCP/fp8KV、补丁队列、镜像与脚本自伤）

### DCP 与长上下文怎么起（launcher 没有 DCP 开关，走直通口）
- `VLLM_EXTRA_ARGS="--decode-context-parallel-size 8 --dcp-comm-backend ag_rs"`（上游只验证过 `ag_rs`）。
- 叠 fp8 KV：`--kv-cache-dtype fp8_e4m3`（只改 KV **存储**、无需 FP8 矩阵核；本机配置层已实测接受）。
- **KV 算术（本机唯一该用的口径）**：DCP=1 → 93.4 KiB/token（8.17 GiB = 94,016 tokens）；
  DCP=8 → **11.5 KiB/token**（5.89 GiB = 534,784 tokens，但权重 +2.11 GiB/rank）。
  1M 单序列需 1,048,576 tokens ⇒ **必须 fp8 KV × DCP=8 才过线**（bf16 KV 只有 534,784，差一半）。
- DCP=8 在短上下文（ISL≈700）是 **−28…−30%**，**必须在长档判**：`TPS_ISL_MULT=24`（≈17k）。

### 补丁队列（quark-int8/dcp_patches）自洽检查
- `python3 quark-int8/verify_patches.py`：①base+队列 与线上树**逐字节**相等 ②GEMV 四副本哈希
  ③split-K 默认 0 ④QR 赋值行未注释。
- 坑：`patch -i <相对路径>` 的路径按 **cwd** 解析 ⇒ 报"找不到补丁文件"、全 rc=2（曾被误归因为
  "多文件 hunk"）。必须传 `resolve()` 后的绝对路径。
- 树上还有 5 个文件被改过而 `base/` **无原件**（`weight_utils.py` + `models/deepseek_v41` 三个 +
  一个）⇒ 本门对它们**无判别力**，属已知盲区，别当成"已覆盖"。
- 新改动一律**追加**（如 0007/0008/0009），**不要重生成旧片**：重生成会让"队列==树"构造性为真，
  自检从此失去判别力，还会把后补的修复静默并进旧片、改写归属。生成+自证脚本：
  `quark-int8/dcp_patches/make_patch789.py`（生成后必须自证 sha256 相等）。

### Docker 自建镜像的坑（QR 镜像就是这么做的，也是这么踩的）
- `docker commit` 会把**被 commit 容器的 Entrypoint/Cmd 一并固化**。用 `--entrypoint bash` 起的
  bake 容器 commit 出来入口就是 `bash -c sleep …`，launcher 传的参数会被当脚本执行 —— 表现为
  容器**秒死**、日志只有一行 `/models: Is a directory`。正确做法显式还原并在事后逐行核对：
  `docker commit --change 'ENTRYPOINT ["vllm","serve"]' --change 'CMD ["bash"]' --change 'WORKDIR /app' <ctr> <img>`
  `docker inspect -f '{{.Config.Entrypoint}} {{.Config.Cmd}} {{.Config.WorkingDir}}' <img>`

### 数字口径（本机最容易记错账的地方）
- **请求级吞吐 ≠ 纯 decode tok/s**：`qr_tps_probe.py` 量的是含 prefill 的请求级（conc32 纪录 123.78），
  旧报告的 9.60–10.71 是纯 decode 口径，**两者不可互比**；只可同一把尺子内横比。
- 报告单流 TPS **必须带 ctx**：ctx≈800 约 10 tok/s，ctx=8192 约 4 tok/s。
- **内核倍数 ≠ 端到端**：split-K 内核 7.2×，端到端 −13%。

### 脚本自伤清单（今天为此烧掉两个 GPU 窗口，逐条都真发生过）
- `pkill -f <模式>` 会匹配**执行它的 shell 自己**（把自己杀掉）；`ps -eo args | grep -F '…api_server'`
  会匹配 **grep 自己** ⇒ 门永远不过。用 pgrep + 方括号模式 `api[_]server` 自保。
- `[ "$x" -ge $${VAR:-1} ]` 里的 `$$` 是 **PID** ⇒ 整数比较报错、超时判断整体失效。
- `bash -c` 里引用父 shell 变量必须 **export**，否则静默变空串（结果表写进空记录就是这来的）。
- 判死/删容器**之前先存日志**，否则现场消失；就绪判定必须**同时看容器 State**，否则会为秒死的
  容器空等满超时（曾空等 45 分钟，fail-fast 后单次失败反馈约 100 秒）。
