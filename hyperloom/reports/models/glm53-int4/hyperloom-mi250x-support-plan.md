# 让 Hyperloom 正式支持 MI250X：改动清单与验证方式（待审）

**状态**：方案文档，**未实施**。依据全部来自本机实测/源码阅读（每条给文件:行号）。
**前置**：当前 GLM-5.3 的 12 小时 optimize 会话正在跑（baseline 的 vLLM 已起，8 卡各 ~54.7 GiB），
这些改动要等它结束再做，否则会话 provenance 与磁盘上的表不一致。

## 0. 结论先行

真正必须动的只有 **3 个文件的 3 处**（身份行 / runner 映射 / roofline 常数），加上 **KB 数据行**；
**不需要 fork Magpie**——前提是采纳下面第 2 项的"runner 映射到 mi300x"方案。

## 1. 身份行 —— `hyperloom/common/gpu_identity.py`（1 行）

    AMD_GPU_DISPATCH_IDENTITIES["mi250x"] = ("gfx90a", 110)

- 表项语义 = **(dispatch gfx arch, CU 数)**；CU 数的消费方实测三处：
  `common/gpu_partition.py:404`（`cu_total = identity[1]`，算 compute partition）、
  `orchestrator/kernel/_kernel_decisions.py:1367`（守卫）、`breakdown/.../assembler.py:421`（标签）。
- **必须"每设备"口径**：mi300x 的 304 是整块 OAM（= 一个 torch device）；本机 8 个 GCD 各自是一个 device
  （`torch.cuda.device_count() == 8`），所以取 **110（每 GCD）**，不是 220。
- 这一行顺带让 `--gpu-type mi250x` 变合法：choices 由该表派生（`cli/parser.py:264`）。
  当前实测报错就是这一条：`invalid choice: mi250x (choose from mi300x, mi308x, mi325x, mi355x)`。

## 2. runner 标签 —— `hyperloom/inference_optimizer/gpu_types.py`（1 行）

- **推荐**：`_GFX_TO_RUNNER["gfx90a"] = "mi300x"`（即 gpu_type=mi250x 但 runner 仍复用 mi300x 的通用脚本）。
  依据：`Magpie/scripts/benchmark/vllm_mi300x.sh` 只有 159 行、**无任何 arch 分支**，是通用 vLLM harness；
  唯一的硬件强相关是 `VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-1}`（第 71 行）与 `--gpu-memory-utilization 0.95`，
  而容器 env 已经是 `VLLM_ROCM_USE_AITER=0` / `VLLM_ROCM_USE_AITER_MOE=0` ⇒ 默认值不会生效。
- 备选（上游更干净但工程量大）：映射到新标签 `mi250x` ⇒ 必须在 Magpie 的 `scripts/benchmark/` 新增 `vllm_mi250x.sh`。
  注意 `Magpie/modes/benchmark/benchmarker.py::_prepare_benchmark_scripts()` 会**覆盖式**把 Magpie 的脚本拷进
  `<InferenceX>/benchmarks/`，所以"只往 InferenceX 目录塞一个文件"会在重克隆/重钉时丢失 ⇒ 必须走 patch/overlay。
- ⚠️ **不要**用 `--benchmark-scripts-dir` 来外挂 runner 脚本：该 flag 映射到 `HYPERLOOM_BYPASS_SCRIPTS_DIR`，
  是"操作者自带 entrypoint 脚本"的**另一个机制**（`bypass_scriptable.py:14/74`、`_workload_envs.py:465`）。

## 3. roofline 常数 —— `hyperloom/orchestrator/kernel/roofline_ceiling.py`（~10 行）

    _MI250X_PEAK_TFLOPS = {"fp32": <实测>, "bf16": <实测>, "fp16": <实测>}   # gfx90a 无 fp8/fp4，不填
    HW_SPECS["mi250x"] = {"hbm_gb": 64.0, "hbm_bw_gbps": 1638.0, "peak_tflops": _MI250X_PEAK_TFLOPS}

- **口径必须与 num_gpus 一致**：`bw_total = spec["hbm_bw_gbps"] × num_gpus`，而 `num_gpus` 的定义是
  **tensor-parallel degree**（`roofline_ceiling.py:1173` 注释）＝本机 8 个 GCD ⇒ 取**每 GCD** 的 1638 GB/s 与 64 GiB，
  而不是整卡 3277/128（否则数字差 2×）。
- 不加这张表的后果（实测代码路径）：`_resolve_peak_tflops` 返回 0.0 走降级，而 `_memory_bound_*` 直接 `return 0.0`
  （`roofline_ceiling.py:1191-1193`）⇒ **roofline 静默消失**。所以第 3 项不是可选项。
- `peak_tflops` 建议**实测标定**而非抄手册：FP32 峰值 = 110 CU × 64 lane × 2 × 实测时钟（`rocm-smi --showclocks`）；
  矩阵峰值用一次 MFMA 微基准（可复用本仓 `hyperloom/kernels/` 的写法），并在注释里写明来源。

## 4. KB 配方行（数据，不是代码）

- 现状：KB 里**已经有 mi250x 键**（`kb/deepseek-v4.1-flash-gguf/mi250x/…`、`kb/ornith-1.5-397b-fp8/mi250x/…`），
  本仓 `scripts/seed_recipe_kb.py` 里也写着 `HARDWARE = "mi250x"`；只有 `kb/glm-5.3-ct-int4-w4a16/mi300x/…` 挂在 mi300x 下。
- 改动：让 seed 为 GLM-5.3 也发一份 `…/glm-5.3-ct-int4-w4a16/mi250x/…`（7 元组目录契约不变），
  然后 `python3 scripts/seed_recipe_kb.py && python3 scripts/verify_recipe_kb.py`。
- 说明：KB 行是"种子网格/先验"，**缺行不致命**（实测日志 `target_analysis … status=skipped reason=model_mapping_miss rows=0`
  是跳过而非失败），但会明显影响候选生成质量。

## 5. 工程约定：第三方代码必须走 patch

Hyperloom / Magpie 都是**第三方、不入库**（CLAUDE.md §4/§6）。上面 1/2/3 项落地方式：
新增 `hyperloom/patches-local/hyperloom-mi250x-identity.patch`（含三处 diff），在 `DEPENDENCIES.md` 记一行，
由 `scripts/bootstrap.sh` 还原 ⇒ 否则下次 bootstrap 会被覆盖。

## 6. 验证方式（V1–V6，从小到大）

| # | 验证 | 判据 | 成本 |
|---|---|---|---|
| V1 | 静态断言：`mi250x` 在 parser choices；`gfx90a→runner` 符合预期；`HW_SPECS["mi250x"]` 存在且 `_memory_bound_decode_tput(gpu_type="mi250x", num_gpus=8) > 0` | 三条全过 | 秒级、不需卡 |
| V2 | CLI 接受性：`optimize --gpu-type mi250x --model /tmp/nope` | 报错变成模型路径相关，**不再**是 invalid choice（这正是本次证伪用的手法） | 秒级 |
| V3 | runner 解析：极小预算跑一轮（`--max-ticks 1` 或 `--max-hours 0.05`），日志断言 `Using Magpie generic script: benchmarks/vllm_mi300x.sh` 且 baseline 起服务成功（显存上去 + 出现 `GPU KV cache size`） | 两条都出现 | ~10 分钟 |
| V4 | **roofline 对拍**（关键，防"更好看但更假"）：把 `T_mem` 与已实测并列 —— 权重 1.47 GB/token/rank、单流 ~98 ms/步（ctx≈800）、10 tok/s、纯 load 620–898 GB/s | MI250X 口径的 `T_mem` 应比 MI300X 口径低约 3×，且与实测同量级；若加表后数值与 mi300x 完全一致 ⇒ 说明没查到表（假通过） | 分钟级 |
| V5 | KB：`seed_recipe_kb.py` + `verify_recipe_kb.py` 全绿；identity 串含 `mi250x`；查询日志不再 `rows=0` | 全绿 | 分钟级 |
| V6 | 端到端小模型：本仓 `models/` 里已有的 tiny Qwen2.5-0.5B，`--max-hours 0.2 --tp 1 --gpu-type mi250x` | baseline→explore 全链路无 gpu_type 相关报错 | ~15 分钟 |

## 7. 风险与边界（诚实）

- **gpu_type 修对 ≠ 官方支持**：基准脚本默认值、结果门限、`target-roofline` 门控都是按 MI300X/MI355X 标定的，
  即便口径正确，部分门控仍需重标；这属于后续项。
- gfx90a **无原生 FP8/FP4、AITER MoE 路径不可用** ⇒ 任何默认开 AITER 的脚本都必须显式关掉
  （本容器已 `VLLM_ROCM_USE_AITER=0`、`VLLM_ROCM_USE_AITER_MOE=0` 实测生效）。
- 双 die 语义：MI250X 一个 OAM 两个 GCD，Hyperloom 侧把每个 GCD 当一个 device（与 `num_gpus=TP` 一致）；
  若将来做单卡 TP1 场景，需要再确认 CU/BW 口径是否仍按 GCD 计。

## 8. 建议实施顺序

1. 三个文件的三处改动 + V1/V2（几分钟，零风险）；
2. V6 小模型端到端 + V3/V4（确认 runner 与 roofline 真的生效）；
3. KB 行 + V5；
4. （可选）若要把 runner 也做成真 `mi250x` 标签，再补 Magpie 的 `vllm_mi250x.sh` patch。
---

## 9. 实施记录（2026-09-21 19:0x，**已实施并验证**）

### 审计阶段改掉的两处（原方案有错）

1. **CU 数不是 110 而是 104**：`torch.cuda.get_device_properties(0).multi_processor_count == 104`（本卡 SKU
   `AMD Instinct MI250X / MI250`，Card SKU D65210V）。厂商 47.9 TFLOP/s per OAM 是按 110 CU 计的，
   属另一 SKU。**连带修正**：因为 104 = 8x13，`gpu_partition` 的 `cu_total % partitions == 0` 对 **8 分区（CPX）成立**
   （原方案按 110 推出的"CPX 会失败"是错的；110 才不整除 8）。16/32 分区仍不行。
2. **真正的 runner 折叠点是 `_gpu_runner_type()` 而不是 `_GFX_TO_RUNNER`**：前者才是 state.gpu_type →
   Magpie runner 标签的规范化函数（`cli/__init__.py:2177/2463`、`conc_sweep.py:1336` 调用）。
   实际改动：`if normalized in ("mi325x", "mi308x", "mi250x")` ⇒ 折叠到 mi300x。
   `_GFX_TO_RUNNER["gfx90a"] = "mi250x"` 也加了，但那是 torch 探测兜底路径（`_autodetect_gpu_type` 的第二条路）。

### 落地的改动（3 个文件 4 处）

| 文件 | 改动 |
|---|---|
| `hyperloom/common/gpu_identity.py` | `"mi250x": ("gfx90a", 104)`（含来源注释） |
| `hyperloom/inference_optimizer/gpu_types.py` | `_gpu_runner_type` 折叠 mi250x→mi300x；`_GFX_TO_RUNNER["gfx90a"]="mi250x"` |
| `hyperloom/orchestrator/kernel/roofline_ceiling.py` | `_MI250X_PEAK_TFLOPS`（fp32 22.6 / bf16·fp16 181.0，按 104 CU×64×2×1.7GHz 与 CDNA2 的 8x MFMA 倍率推导）+ `HW_SPECS["mi250x"]`（每 GCD：64 GiB / 1638 GB/s） |

固化方式：`hyperloom/patches-local/apply_mi250x_identity.py`（**幂等 applier**，第三方代码不入库；无 pristine 树可 diff，
故以"缺什么补什么、可重复执行"等价表达 patch）。已实测可重复执行（第二次全部报 already）。

### 验证结果

| 项 | 结果 |
|---|---|
| V1 静态断言（6 条 + 自动探测） | **ALL PASS**；额外收获：`rocm-smi` 自动探测现在直接返回 `mi250x`（产品名含 "MI250X"）⇒ 不传 flag 也对 |
| V1 数值口径 | T_mem: mi250x 13.1 TB/s vs mi300x 42.4 TB/s ⇒ **比值 0.309（低 3.2x）**；与实测单 GCD 纯 load 620–898 GB/s（每 GCD 峰值的 38–55%）自洽 |
| V2 CLI 接受性 | `--gpu-type mi250x` 被接受，报错从 "invalid choice" 变成环境变量缺失（隔离测试未 source .env，符合判据） |
| IR-2 | `INSTALL_RC=0`，且**三处改动未被 install.sh 覆盖**（已复验） |
| IR-1 | PASS（8 卡各 10 MiB） |
| 上线复验 | 新会话日志出现 `gpu_type=mi250x`，runner 折叠为 mi300x，baseline 正常起服务 |

### 尚未做（按需）

- V3–V6 的完整版（等这次 12h 跑出 baseline 数字后对拍 roofline；小模型端到端 smoke）。
- KB 的 `glm-5.3-ct-int4-w4a16/mi250x/` 配方行（用 `scripts/seed_recipe_kb.py` 生成 + `verify_recipe_kb.py` 校验）。
- 可选的"上游更干净"路线：给 Magpie 新增 `vllm_mi250x.sh` 并把 mi250x 加进 `MAGPIE_BUILTIN_SCRIPTS`，
  再把 `_gpu_runner_type` 的折叠去掉。
