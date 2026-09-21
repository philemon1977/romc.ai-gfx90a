# T2 · 模型 × 量化层

> 同一模型的 bf16 / fp8 / int8 / int4 / GGUF 档位各有各的事实，**不可互推**。

<!-- ── 搬运自 SKILL.md L582-669 ── -->
> **T2 · 模型 × 量化层（GLM-5.3 自转 int4）** — 臂 8121 坐标、转换规程、MoE 调优表 device_name 陷阱、转换侧工具索引。

## 本会话新增（2026-09-21：GLM-5.3 自转 int4 权重的转换规程 + 臂 8121 坐标 + 调优表 device_name 陷阱）

来源：`GLM-5.3-CT-Int4-W4A16`（**本机自转**，与上面 DSV4.1 线是两套独立证据）。
转换全过程：`$AI/docs/GLM-5.3-INT4-量化与-gfx90a-加速-记录-2026-09-20.md`（+ 交接 `GLM-5.3-交接-2026-09-20.md`）。
以下三块此前在本技能 **0 命中**。

### 臂 8121 的实物坐标（Serving arms 表里那一行，这里给路径与判据）
- 产物 `/mnt/kioxia-cm6-3t8/ai/models/ZhipuAI/GLM-5.3-CT-Int4-W4A16` = **402 GB / 141 分片 /
  401.3 GiB / 177,173 张量**（审计 PASS）；源 FP8 `/mnt/kioxia-cm6-3t8/ai/models/ZhipuAI/GLM-5.3`（704 GB）。
- 结构：`glm_moe_dsa` → vLLM 走 `vllm.models.deepseek_v32`（**不是** `deepseek_v41`）⇒ DSV4.1 的
  `models/deepseek_v41/*` 补丁对本臂**不适用**；78 层 = 3 dense + 75 稀疏 MoE（+1 MTP），256 专家 top-8。
- 启动器 `$AI/models/ZhipuAI/launcher/glm53_int4w4a16_vllm_rocmnightly0918_32k_8121_mi250dx8.sh`
  （`PORT=8121`、`IMAGE=vllm/vllm-openai-rocm:nightly-0918`）。它自带三条硬门：有别的 api_server /
  自己的同名容器没清干净 / 任一 GCD >5 GiB ⇒ 直接拒绝起服，**不要绕**。
- 尺寸口径：TP8 权重 **50.84 GiB/rank**（DCP=1；DCP=8 时 52.95，MLA 投影按 rank 复制）；32K 档 KV 预算
  **8.17 GiB = 93,776–94,016 tokens**（多次 boot 两个值）——与「DCP 与长上下文」一节同一把尺子。
- ⚠️ **配方库缺口（2026-09-21 已补）**：`$AI/docs/recipes/serving/` 原先没有 8121，现补了
  `8121-glm-5.3-ct-int4-w4a16-vllm-tp8.md`（12 条 `defaults:` 反漂移断言已过审）+ 配对环境配方
  `vllm-openai-rocm-nightly-0918.md`，并在 `config/ports.conf` 登记端口：`audit_recipes.py` 60→59 项、
  `audit_ports.py` 的「端口 8121 未在 ports.conf 登记」消失（两闸口仍因**其它臂**为红，与本臂无关）。
- ⚠️ **配方 → 技能数据/KB 的链路在「抽取」步是断的**：`data/*.json` 与 Hyperloom KB 都由
  `.tmp/kb_extract/serving_*.json` 喂，而那份抽取**停在 2026-09-15/16**（今天新增的配方一条都没进去）
  ⇒ 改了配方别以为技能数据会自动更新；要么重做抽取，要么手写进本文件（本会话走的就是后者）。

### 自产 checkpoint 的转换规程（FP8 源 → compressed-tensors int4 W4A16）

**先算账再动手**（本机可用 HBM ≈ 496 GiB）：源是 FP8 + 128×128 块 scale、**753.3 G 参数**、694 GiB ⇒
全 int8 **701.6 GiB 装不下**；方案 C（专家+注意力+indexer int4，共享专家/路由/norm/嵌入 bf16）
预测 **402.2 GiB** / 实测 401.3 GiB。**为什么不是 Quark**：镜像自带 amd-quark，其 int8 scheme 在 gfx90a
是有内核的（CK int8），但 int8 装不下；其 int4 走 OCP MX（MXFP4），源码注释原文
`All OCP MX schemes (W4A16, W4A8, etc.)` ⇒ **要 fp4 硬件，CDNA2 没有**。⇒ 选 compressed-tensors W4A16
（Triton W4A16 线性 + WNA16 MoE + 本机 GEMV 补丁，全部落在 MFMA 上）。

**四个「看起来跑通、其实产物全错/不全」的 bug**（各留过现场）：

| 症状 | 根因 | 处置 |
|---|---|---|
| plan 说 500 个 fp8_block，产物 0 个 `weight_packed` | `modules_of` 不认 `.weight_scale_inv`（DSV4.1 叫 `.scale`）⇒ scale 被当独立模块、量化分支因 `"scale" not in leaves` 整体退回 keep | 让 `modules_of` 认 `.weight_scale_inv` |
| 形状与数量级同时错 | GLM 的块 scale 是 **F32 乘数**，不是 DSV4.1 的 E8M0 指数字节 | 判决尺：`w×scale_inv` 的 \|max\|=0.083（✓）vs `w/scale_inv`=3.4e6（✗） |
| `576/5=115.2` 不是整数 | 块高固定 128，**最后一块不满** | `ceil(n/128)` + 截断展开，不要 `n // scale.shape[0]` |
| 行分块与 scale 下标错位 | `s[r0:r1]`：576 行权重只有 5 行 scale | 改 `s[r0//128 : ceil(r1/128)]` |

**两条分类陷阱**：① `*mlp.gate*` 会连 `mlp.gate_proj` 一起吃掉（必须写 `*mlp.gate`）；
② 融合模块（如 `indexer.wk_weights_proj`）只能一种 scheme，混排会在服务端以 `KeyError: …weight_packed`
或装载失败暴露。**收口闸口** `quark-int8/audit_glm53_ct.py` 除「该量化的都量化、该保留的都保留」外，
还有一条 **[9] classify 漂移检查**——用 vLLM 自己的 `should_ignore_layer` 反查 ignore 集与量化集是否
相交（相交 ⇒ 服务端按未量化构建 ⇒ 装载失败）。**独立尺子** `verify_glm53_int4.py`：不复用转换器函数，
独立按 CT 约定反量化后与源对拍 ⇒ cos 0.9952–0.9954、max\|err\|/amax 0.0746–0.0747（≈ 对称 int4 理论界 1/14）。
**代价量级**：≈13.8 s/分片 × 141 ≈ 32 分钟（同任务纯 CPU 33 s/分片），`--skip-existing` 可断点续跑。

### MoE 调优表：`device_name` 是变量，不是常量（更正一条既有配方）

`knobs/vllm-fused-moe-tile-seeds.md` §3 把本机卡名口径写成了常量 `AMD_Instinct_MI250X_MI250`。
**同一台机器上实测出现过两个值**：

| 值 | 出处（原文日志） |
|---|---|
| `AMD_Instinct_MI250X_MI250` | 2026-09-17 Ornith 8116，`fused_moe.py:1148`：`Using configuration from …/quark-int8/moe_tuned/E=512,N=128,device_name=AMD_Instinct_MI250X_MI250,dtype=int4_w4a16.json for MoE layer` |
| `AMD_INSTINCT_MI250_(MCM)_OAM_AC_MBA` | 2026-09-20 GLM int4 8121（nightly-0918），`fused_moe.py:1163`：`Config file not found at /moe-tuned/E=256,N=256,device_name=AMD_INSTINCT_MI250_(MCM)_OAM_AC_MBA,dtype=int4_w4a16.json` |

取值路径已核实：`vllm/utils/platform_utils.py:69` = `re.sub(r"[\s/]+", "_", get_device_name())`；
ROCm 侧 `platforms/rocm.py:827` 先查 `_ROCM_DEVICE_ID_NAME_MAP`，本机 `device_id=0x740c`（sysfs
`PCI_ID=1002:740C`）**不在表内**（表里只有 MI300A/X、MI308X、MI325X、RX7900XTX、RDNA3.5、Navi48）
⇒ 回落 `asic_info["market_name"]`（与 sysfs `product_name` 同串）。两次实测的 `fused_moe.py` 行号不同
⇒ 是**不同构建**的取值，别把任一个当「本机常量」。

⇒ 铁律：**表名只能由该臂镜像里的 `get_config_file_name()` 现场生成**（`quark-int8/moe_tune_w4a16.py`
就是这么做的，并拒绝手抄）；**手抄的表名跨臂/跨镜像会静默失效**——现场只有一行
`warning_once "Using default MoE config"`，不报错。两条同源事实：`fused_experts_impl` **没有 `config=`
形参**（注入只能 monkey-patch `try_get_optimal_moe_config`）；M 命中是**最近邻桶**
（`configs[min(keys, key=|x-M|)]`）⇒ 桶要覆盖真实 M 分布。**「挂载≠命中」的实例**：本臂（E=256 / N=256）
在 2026-09-20 的日志里就是这一行 `Config file not found at /moe-tuned/E=256,N=256,device_name=AMD_INSTINCT_MI250_(MCM)_OAM_AC_MBA,dtype=int4_w4a16.json`
——目录挂上了、表却一张也没命中；而 `$AI/config/moe-tuned/` 与 `quark-int8/moe_tuned/` 现存的表还带着旧构建的名字
（且形状是别的臂的 E=512/N=128）⇒ **别据此说「本臂的表已接上」**。
**基线状态**：`moe_tune_w4a16.py` 的基准夹具与生产形状不一致（uint8 `[N,K/2]` vs 生产 int32 `[N,K/8]`
+ bf16 scale）⇒ **它的胜负数字目前不能用于接线**，夹具修好前不要引用。

### 转换侧工具索引（都在 `quark-int8/`；除注明外纯 CPU 只读）

| 工具 | 用途 |
|---|---|
| `convert_glm53_ct_int4.py` | GLM 转换器（GPU；`--skip-existing` 续跑、`--min-free-gib` 合租守卫） |
| `audit_glm53_ct.py` | 完整性 + classify 漂移审计（CPU） |
| `verify_glm53_int4.py` / `verify_glm53_bf16.py` | 独立反量化与源对拍（CPU） |
| `moe_tune_w4a16.py` | MoE tile 实测调优（GPU；夹具待修，见上） |
| `glm53_sizing.py` / `glm53_kv_math.py` | 体量与 KV 算术（「先算账再动手」的执行体） |
| `gpu_gate.sh` / `stack_probe.sh` | 起服硬门与单臂探针（见第二轮增补） |


<!-- ── 搬运自 SKILL.md L670-726 ── -->
> **T2 · 模型 × 量化层（Ornith-1.5-397B 四线）** — INT8 / MXFP4 / CT-Int4 定案，int4 自建 aiter 内核的可行性边界。

## 本会话新增（2026-09-21 第四轮：Ornith-1.5-397B 量化三线定案 + int4 自建 aiter 内核可行性）

来源：Ornith-1.5-397B 的 Quark INT8 / MXFP4 / CT-Int4 三线在本机的端到端实测。
证据：`quark-int8/RESULT.md`、`quark-int8/INT4_CT_PLAN.md`、`EVIDENCE_real_int8.txt`、
`EVIDENCE_v2_aiter.txt`、`EVIDENCE_v3_mxfp4_mtp.txt`（数值都带出处）。

### 1. Ornith 量化四条线（v1–v4 均已 TP8 起服实测；起服按 §起服规定申请）

| 版本 | 产物 | 大小 | 权重/rank | 单流 decode | NLL(ppl) | KV 池 |
|---|---|---|---|---|---|---|
| v1 INT8 **仅 experts**（Quark file2file） | `Ornith-1.5-397B-Quark-Int8` | 421 GB | 52.6 GiB | 37.57 tok/s | **1.9687**(7.16) | 331,954 |
| v2 = v1 + attn/GDN 投影也 INT8 | `…-Quark-Int8-Attn` | 414 GB | 51.8 GiB | 34.39 tok/s | 2.0239(7.57) | 614,518 |
| v3 MXFP4 W4A16 | `…-Quark-MXFP4-W4A16` | 239 GB | ~30 GiB | **7.16 tok/s** | 1.9868(7.29) | 1,024,455 |
| **v4 CT-Int4 W4A16 + MTP(5) @256K**（8116 臂） | `…-CT-Int4-W4A16` | **223 GB** | ~27 GiB | **68.83 tok/s**（5 次中位数） | 1.9558（ngram 档） | 1,555,665 |

- v4 的权重由本会话的 `pack_int4_ct.py` 产出（2026-09-16 23:26–23:45，单卡 cuda，**18.9 min**：`split 61440 fused expert tensors; quantized 92160 2D weights`，gs=128、仅 experts、`ignore` 12 条）；
  之后的起服与调参（MTP 深度扫描）由 8116 臂完成，**单流最好成绩即出自它**，细节与口径见 `quark-int8/RESULT.md §7`。
- **v4 为何比 v1 快 1.83×**：主因不是 4bit 内核，而是 **MTP 深度**。§7 的步长模型 `步长 = 27.0 ms + 9.64 ms × tokens/步`
  ⇒ **61% 的步时间与 token 数无关**（TP8 每步约 120 次 allreduce + 发射间隙 + 小 M 效率），MTP 摊薄的正是这一项。
  **推论（对"要不要自研 int4 内核"直接相关）：单流上换内核/调 tile 的天花板受同一个固定项约束。**
- **v2 的教训：把注意力/GDN 也拉进 INT8 不加分**（−8.5% 吞吐、NLL +2.8%）。默认只量化 routed experts。
- **v3 的教训（vLLM 0.29 源码）**：ROCm 上 MXFP4 MoE 只有 `AITER_MXFP4_BF16`（被 CDNA2 门控禁用）+ `EMULATION`，**没有 triton 回落** ⇒ 逐 token 反量化 ⇒ 慢 5 倍。**本机 MXFP4 = 省显存不省算力，别拿它换吞吐。**
- INT8 线性走 aiter 的日志证据：`Selected AiterInt8ScaledMMLinearKernel for QuarkW8A8Int8` + `[aiter] import [module_gemm_a8w8]`（依赖上一节 JIT 缓存 / `AITER_JIT_DIR`）。
- 量化侧：Quark 官方流式 `ModelQuantizer(QConfig).direct_quantize_checkpoint(...)`（逐分片、**免校准** W8A8-dynamic）+ `LLMTemplate.get("qwen3_5_moe")` 的 exclude 与 `SplitFusedExperts` 融合专家拆分（`quantize_ornith_int8.py --coverage experts|experts_attn --weight-dtype int8|mxfp4`）。**`--device cuda` 比 `--device cpu` 快 4–8×**（同配方 MXFP4：CPU ~2 h vs GPU **14.8 min**；INT8 CPU 68.6 min）。
- 单流数字的口径纪律（§7 实测）：臂内离散 **12–24%** ⇒ 必须**同服多请求中位数**；且 TPS 强依赖 prompt 可预测性（同一服务换 prompt **56 → 88 t/s，1.57×**）。

### 2. int4 只能走 compressed-tensors，Quark 到不了（实测否决）

- `QuarkConfig` 对 `int4/uint4`（per_group / per_channel）一律 `NotImplementedError: No quark compatible scheme was found`；同代码路径的 mxfp4 基线可解析为 `QuarkOCP_MX`（A/B 对照见 `INT4_CT_PLAN.md`）。
- W4A16 的消费者是 `auto_awq / auto_gptq / compressed_tensors_moe_wna16 / moe_wna16`；`_POSSIBLE_KERNELS[ROCM]` 含 `TritonW4A16LinearKernel`（MoE 侧 `int_wna16` 后端 = `MARLIN/BATCHED_MARLIN/TRITON`）。
- 产出工具 `quark-int8/pack_int4_ct.py`：流式 CT `pack-quantized`（`weight_packed` int32 `[N,K/8]` + `weight_scale` fp32 `[N,K/gs]` + `weight_shape` int64；组沿 K；对称 int4 RTN，免校准）。数值 normRMSE **0.117** / cosine **0.9932**（与 MXFP4 的 0.112/0.9937 同级）。
- ✂️ **静默 bug（已修 + 已加断言）**：融合专家张量名 `...mlp.experts.gate_up_proj` **没有 `.weight` 后缀**，用 `core.endswith()` 判断会**永不命中** ⇒ 98% 权重被静默跳过。现按 `name` 匹配并在结尾断言 `quantized_tensors>0`。v4 打包日志：`split 61440 fused expert tensors; quantized 92160 2D weights`（单卡 18.9 min）。

### 3. int4 自建 aiter 内核？——**现有 aiter 源码做不到；要立项就得新写，且收益上限受实测约束**

- aiter 资产普查（纯文件读取，不 import）：`module_gemm_a4w4_asm.so` / `module_gemm_a4w4_blockscale.so` / `module_moe_mxfp4_aux.so` 都在，但 int4 家族源码里 arch 字符串 **gfx950 ×1990 / gfx942 ×3 / gfx90a ×0**（`configs/model_configs/*a16w4*`、`*a4w4*` 亦然）⇒ 它们面向 CDNA4 的 fp4 scaled-MFMA。**a8w8 那条"通用 CK 模板 JIT 重编到 gfx90a"的路子对 int4 不存在**——a8w8 能成是因为 CDNA2 有原生 int8 MFMA。
- 可复用的是**基础设施**而非内核：`AITER_JIT_DIR` 离线缓存（`~/.cache/aiter-gfx90a`）、CK 实例生成、pybind ABI 补丁、tuned-CSV 机制。
- 收益上限的实测约束（两条独立证据）：① 本机 GLM-5.3-CT-Int4 归属表里 `triton_w4a16_gemm` 占 decode 一步 **18.9%**，自研 MoE GEMV 优化后只剩 **1.8%**（内核 8.7×/5.0× 仍被撤回），且「慢不是带宽、是层内串行与启动延迟」；② Ornith 自己的 int4 臂已经不用任何新内核就拿到 **68.83 tok/s**（`RESULT.md §7`），其步长模型显示 **61% 的步时间与 token 数无关** ⇒ 单流上换 GEMM 内核的天花板被同一个固定项封住。
- 结论：**先做这两件（都不是内核）** —— ① 补装 gfx90a 的 a8w8 调优表（见 §5，纯文件操作）；② 在 MTP 深度/通信/固定开销上继续榨（§7 已证 +69% 来自 MTP 深度）。只有 profile 归属里 int4 GEMM 明显占主导（>25–30%）时，才值得考虑下面这条立项路线。
- 若真立项，唯一可能赢的设计是 **W4A8-int8**（int4 权重在寄存器 lift 成 int8 + 原生 int8 MFMA + 组内 requant，上游无源码，需新写），**前置条件**是先拿到 profile 归属证明 GEMM 是瓶颈（取窗口方法见上文 §Profile a step）。

### 4. 本会话固化的运维技巧（都已成脚本）

- `quark-int8/make_tiny_real.py`：用 patched transformers 造**同名同构 272 MB 小模型**，同配方 24 s 量化、~2 min 起服 ⇒ 配方排错从「每轮 40 min 加载」压到分钟级。已救两次：`sum(mrope_section)==rotary_dim//2` 断言、融合专家命名 bug。**新配方先过 tiny，再上 397B。**
- `quark-int8/watch_container.sh <容器> <超时> <端口>`：把 `RuntimeError|initialization failed|EngineCore failed|ValueError|TypeError|has no attribute|failed to start|No available memory` 当**终止条件**——只等 `/health` 会为空等满超时。
- `quark-int8/nll_probe.py <port> <model> <label>`：固定文本 `prompt_logprobs` 的平均 token NLL = 量化 A/B 的质量尺子（已积累 1.9687 / 2.0239 / 1.9868）。跨臂只用同一把尺子横比。
- **MTP × 量化的地雷**：开 `--speculative-config method=mtp` 后草稿模型（`Qwen3_5MoeMTP`）专家层是**未量化 bf16**，vLLM 会自动为它选 `ROCm AITER` MoE 后端并崩（`Unquantized MoE backend ROCm AITER does not support the deployment configuration`）⇒ 加 `-e VLLM_ROCM_USE_AITER_MOE=0`（目标模型的量化专家不受影响）。
- 混合 GDN 架构的并发上限：Mamba state cache 与 KV 抢余量，`--max-num-seqs 256` 报 `exceeds available Mamba cache blocks`（Ornith 用 16）；`--gpu-memory-utilization <0.95` 会先报 `No available memory for cache blocks`。

### 5. 已发现但**尚未回收**的便宜杠杆：aiter a8w8 调优表没装进 env

`patches/gfx90a/aiter_a8w8_tuned_gemm_gfx90a.csv` 内有 **54 行 gfx90a**，但三个 env
（`vllm_0.28.0_rocm72` / `vllm_master_rocm724` / `wu1w-int8-028`）里的
`aiter/configs/a8w8_tuned_gemm.csv` 只有 gfx942=26 + gfx950=553 ⇒ **gfx90a 的调优从未生效**，
生产日志里的 `not found tuned config in a8w8_tuned_gemm.csv, will use default config` 就是它。
装表是纯文件操作（CPU），建议在任何 aiter INT8 计时/对比之前先补上。



---

## 附录 D · ⚠️ 版本绑定核实（引用这两份记录前必读）

- `docs/DeepSeek-V4.1-Flash-CT-INT4-W4A16-转换记录-2026-09-18.md`（175 KB / 2462 行）
  **全文只有 nightly 一条线**（`0.29.1rc1.dev47+gdc36fcce9` L3-4 → `nightly-0918` = `dee37d891` L1800/L1838）；
  **`0.28.0`、`master`、`748B`、`GLM-5.3` 在全文 0 命中**。
  适用域：端口 **8119** / 补丁树 `ct_w4a16_dsv41_n0918` / 模型 **DeepSeek-V4.1-Flash** /
  量化 **ct-int4 W4A16（uint4b8, g32）**；**与 GLM-5.3 int4（走 `deepseek_v32`）不是同一套补丁**。
- `docs/MI250X-AITER-INT4-内核复核-2026-09-17.md` **全文无「ROCm 7.2.4」字样**（只隐含在脚本名
  `…_vllm_rocm72_…` 里）。
- ⇒ **版本绑定要按文件内实测字符串写，不能按文件名或印象推**，否则会造出不存在的版本绑定。

## 附录 E · 自转 int4 权重的硬约束与判据（**T2**）

> 出处 `docs/DeepSeek-V4.1-Flash-CT-INT4-W4A16-转换记录-2026-09-18.md`。
> 本节是 §4.1–§4.33（约 1900 行）里**最可复用**的部分——技能此前只吃了 §4.34–§4.39。

- **两条装载硬约束（能装上但结果全错）**〔§4.1 L235-243〕：
  ① 合并列装载的 `shard_offset`/`shard_size` 必须传**未打包单位**
    （loader 内部 `_adjust_shard_indexes_for_packing` 再除 `packed_factor=8`），
    按打包单位传会算成 offset 160（应 0）⇒ **只能靠真实装载器验证**；
  ② `TritonW4A16LinearKernel` 只实现 fp16/bf16 激活，而 `LinearBase` 默认
    `torch.get_default_dtype()` = **fp32** ⇒ 报 `Only float16/bfloat16 activations are supported`
    ⇒ **launcher 必须显式 `--dtype bfloat16`**。
- ★ **TileLang mHC 在 gfx90a 上静默算错 `layer_input`**〔§4.8 L277-325〕：
  `mhc.py::_has_tilelang_mhc()` **只排除 gfx942、未排除 gfx90a**；
  症状是**只有 `layer_input` 非确定性错到 1.5× 量级**，`post_mix/pre_mix/comb` 全对（1.2e-7）、
  logits 尺度形状全正常 ⇒ **无法从输出反推，必须连跑多次**（第 2 次恰好是对的）。
  修法 `if on_gfx90a(): return False` 落回 torch 实现 + launcher fail-closed grep。
  **两条通用规则**：①「上游自己都不信」的平台排除条件，换架构时要**重审放行/排除列表**，
  优先看平台条件分支而非算子数学；② **wave64 是 gfx9 系统性风险面**——
  凡有 `tx<32` / `warpSize` / `__ballot(0xffffffff)` 之类 warp32 假设的融合核，
  在 gfx90a 上一律按「可能静默算错」验。
- ★ **「按参数占比线性外推损伤」是错的**〔§4.19 L877-890〕：
  `wq_a`/`wkv` 只占注意力参数 **7.2%**，却贡献 **4.62%** 输出偏差——因为它们在**最前端**
  （`qr=rmsnorm(wq_a·x)` → `q=wq_b·qr` → softmax），误差穿过 softmax 非线性被放大；
  `wo_b` 在末端不被放大 ⇒ **必须实测**，不能按占比估。
- ★ **KV 格式才是 256K 的瓶颈，不是权重精度**〔§4.24 L1274-1320〕：
  官方 global KV **890 B/token**（FP4 main KV）vs 我们 bf16 压缩 KV ≈**3,660 B/token**
  （= 1.97 GiB ÷ 595,565 实测）≈ **4.1×**。
  另：**两份 `config.json` 别取错**（模型根目录是 HF 风格 `text_config.kv_source_layer_ids`；
  `inference/config.json` 才是参考实现的 `kv_source_layers`）。
- **CT 的 `ignore` 通配匹配的是 vLLM 模块前缀名，不是 checkpoint 张量名**
  （`wq_a`+`wkv` → `attn.fused_wqa_wkv`）⇒ 用 checkpoint 名写 glob 会**静默失效**，
  随后 `KeyError: …weight`。〔§4.10 L417-477〕
- **起服前静态一致性证明**〔§4.22 L1150-1215〕：不变式 = 对**全部模块**断言
  `classify ⇒ 未量化 ⟺ ignore 命中`（**双向**，两个方向都会以 KeyError 炸）；
  逐分片 QA 必须**直读分片 header**，**不能依赖输出目录 index**
  （转换期间它还是试点那份，会把好分片误判成缺失）。
  两条结构事实：mapper 里 **`"mtp.": None` ⇒ MTP/DSpark 权重整条丢弃、根本不加载**；
  共享专家是独立 `DeepseekV4MLP`（两个 Linear 子模块）⇒ **不存在「同模块混 scheme」风险**。
- **逐组最优 int4 scale 的收益按源格式分档**〔§4.16 L682-730〕：
  fp4 路由专家 **1.48–1.61×**、fp8 共享专家/注意力仅 **1.12–1.14×**（平均 1.313×，n=24）；
  bf16 落盘 scale 代价仅 0.007 个百分点 ⇒ 不必改 `scale_dtype`。
  ⚠️ `torch.where(cond,A,B)` 参数顺序写反会把 running-min 变成"更优保留旧值"（收益一度为负）；
  **自检法** = 打印被选值分布，看是否只出现候选集里的值。
- **把量化层改 bf16 不会踩 ROCm 定制 GEMM 快路径**〔§4.19 L923-943〕：
  `amd/rocm.py::_prep` 对 CT 量化模块与未量化模块**都返回 `None`** ⇒ 快路径保持关闭、
  也不会执行 `shuffle_weight` 就地打乱（若被打乱而无可配 scale，结果反而错，被这个 `None` 挡住）。
- **三堵物理墙（含三次 OOM 全误诊）**〔§4.28 L1828-1835〕：
  ① HIP context ≈**1.08 GiB/rank**（worker 建 context 后才测 free）⇒ util 启动门**天花板 0.983**；
  ② profile 期固定 **512 MiB** = `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`（默认 512，
     `rocm_aiter_mla_sparse.py:1061`），**与 len/batch 无关** ⇒ **缩档治不了这个 OOM**；
  ③ 60.95 GiB/rank + 1.08 + 激活≈1.2 + NCCL ≈ 63.98 物理满 ⇒ 过启动门但报
     `No available memory for the cache blocks`。
  另：**prefetch offload（`--offload-group-size`）× 流式装载 = 装载互锁**
  （py-spy 栈 `_load_w13`、26 MiB/s、50 min 无进展）⇒ 该组合不可用；
  但「offload 不可用」**只能限定为 prefetch 后端**（**UVA 后端从未试过**）。
- **短上下文会走上游旁路**〔§4.30 L1893-1910〕：`attention.py:99
  _fill_short_context_topk_indices` 在 `max_seq_len // ratio ≤ topk` 时直接"全选候选"并
  `return None,None,None` ⇒ **稀疏 indexer op 本就不会被调用**。
- **防重复实验**〔§4.33 L2057-2072〕：「注意力全体 bf16」**早已测过且无效**（退化率 77.78%），
  别再花一轮。gfx90a 走纯 torch 回退的**机制原因**：`rocm_aiter_ops.is_enabled()` 被
  `@if_aiter_supported` 包着、要求 `get_cdna_version() > 2` ⇒ gfx90a 恒返回 `None`；
  且该回退含 `int(context_lens[i].item())` = **D2H 同步 ⇒ 图捕获期非法**。
- **选路约束**：ROCm **没有 8bit 线性层内核**（uint8b128 的 exllama/marlin 是 CUDA 专属）
  ⇒ 只能 `triton_w4a16`；CT 的 mxfp4 是 **W4A4**（需原生 fp4 硬件）⇒「CT 仓塞原生 MXFP4」不通。
- **两个量尺纪律**〔§4.15 L626-665〕：① **RMSNorm 输出 rms 精确等于 `*_norm.weight` 的 rms** ⇒
  残差流 rms 0.098→6.85（70×）与 layer15/39 FFN 尖峰被 `ffn_norm.weight` rms（0.02–0.61、随层增长）
  解释掉，是 **checkpoint 自身性质不是运行时缺陷**。

## 附录 F · 上游量化权重的二进制级准入判据

> 出处 `docs/MI250X-三底座量化权重检索-ModelScope-2026-09-05.md`。
> ⚠️ 该文有**两条前提已被实测推翻**（`Qwen3.8-Flash-Next` 小节结论已过期）⇒ **引用前先读其 §00 顶注**。

- 🔑 **Marlin 符号 0 命中**（二进制级证据）：本机 4 个 vLLM `.so` 里 `marlin` 命中 **0**
  （同文件 `scaled_mm=2` / `rocm=676` / `moe=1139` ⇒ 证明 strings 有效、非假阴性）；
  `_rocm_C.abi3.so` 里 weight-only 4bit 只有 `gptq_gemm_rdna3*`（**RDNA3 专用**）；
  Python 侧 `check_moe_marlin_supports_config()` 开头就是
  `if current_platform.is_rocm(): return False`
  ⇒ **AWQ / GPTQ / 任何 compressed-tensors W4A16 的快内核
  （Marlin / Machete / AllSpark / Cutlass-W4A8）在本机全部不存在**。〔§1.1 L48-60〕
- 🔑 **本机 vLLM 上有正经内核的档位只有两种**：**BF16** 与 **INT8 W8A16 gs=128 → Conch**。
  「想要 int4 且快」⇒ 自己产 `group_size=128`（Conch 准入 `gs ∈ [-1,128]`）或直接用 GGUF；
  **不要下 gs=32 的 AWQ/int4**。〔§5.2 L153-154〕
- **三档量化同轮同 die 对拍**（`HIP_VISIBLE_DEVICES=6,7`、同 prompt、`max_tokens=192`、
  warmup 后计时、**350 W cap**）：BF16 **29.52** / INT8-W8A16-Conch **11.79（40%）** /
  INT8 eager 6.41（22%）/ int4-AWQ 12.1（~41%）⇒ **全部落在 BF16 的 22%–41%**。
  ⚠️ **两个百分比分母不同不可混用**（本表以**本次实测 BF16** 为分母；
  `safetensors适配清单` §10 的 11%/56% 以 **HBM 理论上限**为分母）。〔同文 §3 L86-103〕
- **定档必须"以实测吞吐"定**，不以"内核是否准入 / 是否报 fallback / 是否在 GPU 上跑"定。
  〔`docs/MI250X-safetensors模型硬件加速适配清单-2026-09-04.md` §0 L24〕
- **DSV4-Flash 0731 的两道串行硬门**（照抄会撞两次）：先
  `AssertionError: DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, got auto` →
  再 `fatal error: 'rocwmma/rocwmma.hpp' file not found`（**ROCm 7.2.4 树不带 rocwmma**）
  ⇒ 修法 `apt-get install rocwmma-dev7.2.4`；**不能**用 `CPLUS_INCLUDE_PATH` 借 6.4.4 的头。〔§4.2 L122-127〕
- **atom 把 `I8/U8` 无条件映射成 `fp4`** ⇒ 第三方 INT8 checkpoint 会**静默数值损坏**
  （不报错）。〔`docs/MI250X-DSpark-INT8-本机运行记录-2026-09-08.md`〕

## 附录 G · 选型铁律（比任何调参都值钱）

> 出处 `docs/MI250X-vLLM-优化方案.md` §3 S3 L114-116。**已验证是数量级差异。**

1. **无 DSA / 稀疏索引器**；
2. **`head_dim ≤ 256`**；
3. **权重不用 FP8/FP4**（用 bf16 或 W8A8 对称 int8）；
4. **看激活参数量，不看总参数量**（A3B 的 75 t/s vs 27B 稠密 int8 的 8.2 t/s）。

**瓶颈表述**〔§1 L12-23〕：不是带宽/算力/通信/显存，而是**每步的固定成本**
（一个 decode 步串行 ~1178 个细碎 kernel；hipBLASLt 给 M=1 挑了 64×64 tile 的 GEMM）。
2026-09-03 修正：**5 个 bind-mount 的 py 文件**就把 Ornith TP8 **75.0 → 92.7 t/s（+23.6%）**、
TP2 **70.6 → 90.0（+27.5%）**；**QuickReduce 的门是"错误的保守设置"
（gfx90a 数值逐位正确、可用），而 CUSTOM allreduce 的门不能打开。**

⚠️ 这与 `data/knobs.json :: quick-reduce-gfx90a` 的"禁用"判词**并列存在**：
QR 是否可用取决于**该臂的 KV 预算是否容得下那 ~9 GiB/卡固定分配**，不是恒禁。
引用时必须同时给「数值正确性」与「显存账」两把尺子。

**明确不要再做**〔§7.5 L267-270〕：
① 在 gfx90a 打开 vLLM 的 **CUSTOM allreduce**（输出**静默**变 `!!!` 且 GPU 非法访存）；
② 在 QuickReduce 上试 **INT8/INT6/INT4/INT3** 档（CDNA2 无对应指令，且 4 KB 消息本就延迟受限）；
③ 用 **isolated 微基准给"碎核"排序定收益**（同一 topk kernel：孤立 10.2 µs vs in-situ 28.4 µs，
**结论会反**）。

**两阶段 TunableOp**〔§3 S2 L101-113〕（修那 2.4 ms/步的 tile 选错，预期 +10~18%）：
⚠️ **inline 开启与 CUDA graph 捕获相乘爆炸**（9 分钟没能启动）⇒
① `--enforce-eager` 跑一遍开 tuning 并在**退出时** dump CSV；② 生产 `TUNING=0` 只读该表。
本 build **无 `torch.cuda.tunable.write_file()`**，依赖进程退出时 dump ⇒ 需先验证能否落盘。

**两条外部风险**〔§3 S1 L99-100〕：hybrid-GDN 模型上 DFlash 长上下文可能**净亏**
（vLLM issue #54691）⇒ 必须自己扫 ctx 长度；**别开 sleep mode**
（MI250 + spec decode 有崩溃报告 #47548）。

## 附录 H · indexer：bf16 承载比 fp8 存储快 2~3×

> 出处 `docs/MI250X-indexer-非AITER-Triton路径-现状与方案-2026-09-05.md`。
> **适用域：vLLM master 树（`qwen4_exp/amd/`、`glm5next/amd/`）。**

- 🔑 **选型结论：dequant 必须提前到 cache 存储层**（K cache 应存 **bf16**，
  而非照抄上游 fp8e4m3 布局）。〔§3.2 L102-118〕
  `M=1` **1.62 vs 3.22 ms**；`M=2048` prefill **56.5 vs 18.8 TF**（达成率 **31% vs 10%**）。
  根因：**CDNA2 无 fp8→bf16 硬件转换指令**，vendored 内核**在热循环里每次 dot 都重做一遍软件 dequant**。
- 该 kernel 是 **`grid=(M,)` 一 program 一行** ⇒ `M=1` 只有 **1 个 CTA** 在 104 CU 上跑
  ⇒ **必须把 N 方向也切并行**（split-K / 两段归约）。〔§3.2 L115-116〕
- **GLM 走 `forward_cuda` 通用路径**（`index_kpool=4 > 1` 恒成立）⇒
  **解耦不需要 AITER 支持 kpool**；gfx90a 唯一死因是**外层 `is_enabled()` 门**
  （`_aiter_ops.py:157` 用 `get_cdna_version() > 2`）。
  ⇒ **「AITER 不可用」应收窄为「仅 MoE 通路」**：`sparse_attn_indexer_kpool.py:1034/1062`
  的门**不是硬件门**。〔§2 L33-50〕
- **Qwen4Exp 量化红线**〔§4.1 L131-136〕：① KV cache **只能 bf16**
  （`amd/qsa.py` 五处显式 `raise NotImplementedError("…BF16…")`）；
  ② PLE n-gram 表（`320,001,536 × 160`、全表 **102.4 GB**、TP4≈25.6 GB/die、TP1 装不下）
  **绝不能用 FP8 checkpoint**——`_QWEN4_EXP_IGNORED_MISSING_SUFFIXES` 把 `_weight_scale`
  列为可忽略 ⇒ **FP8 PLE 会被裸 cast、scale 被静默丢弃 = 错值不报错**。
- **GLM 起服参数**〔§4.2 L178-179〕：`--kv-cache-dtype` 必须 **auto/bf16**
  （**fp8 会被路由进 AITER decode**——这正是 gfx950 用户照抄
  `--kv-cache-dtype fp8` 到 MI250X 必死的坑）；`--block-size ≥ 128`。
