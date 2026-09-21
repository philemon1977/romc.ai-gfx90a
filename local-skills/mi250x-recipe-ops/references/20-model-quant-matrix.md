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

