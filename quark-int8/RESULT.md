# Ornith-1.5-397B → AMD Quark 原生 INT8（本机 8×MI250X / gfx90a）结果报告

日期：2026-09-16 · 机器：8×AMD Instinct MI250X (gfx90a, 64GB/GCD, 共 512GB) + EPYC 7413 / 251GB RAM
运行时：`vllm/vllm-openai-rocm:nightly`（vLLM 0.29.1rc1.dev47，内置 **amd-quark 0.12.post1**）

## 1. 交付物

| 版本 | 路径 | 大小 | INT8 覆盖范围 | 状态 |
|---|---|---|---|---|
| **v1（官方模板 recipe）** | `/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-Quark-Int8` | **421.0 GB**（123 分片 / 187,124 张量） | 60 层 × 512 routed experts（gate/up/down）= 92,160 个 int8 张量 + 92,160 个 per-channel scale | ✅ TP8 加载 + 真实推理验证 |
| **v2（扩展 recipe）** | `/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-Quark-Int8-Attn` | **414.2 GB**（123 分片 / 187,319 张量） | v1 + `self_attn.{q,k,v,o}_proj` + `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` | ✅ TP8 加载 + 真实推理验证，**INT8 Linear 走 aiter 内核** |

量化格式（两版一致，vLLM `QuarkConfig` 原生识别）：
* weight：`int8` / `per_channel` / `symmetric` / static（输出通道）
* activation：`int8` / `per_channel` / `is_dynamic=true` / symmetric（per-token 动态，**无需校准数据**）
* 保持 BF16：router `mlp.gate*`、`shared_expert*`、`shared_expert_gate`、全部 norm、`conv1d`、`A_log`/`dt_bias`、`linear_attn.in_proj_a/b`(v2 仍排除)、`lm_head`、vision tower、`mtp.*`

## 2. 量化路径（Quark 官方 file2file，不加载整模型）

807GB BF16 装不进 512GB HBM + 251GB RAM，因此使用 Quark 的流式入口
`ModelQuantizer(QConfig).direct_quantize_checkpoint(...)`（逐 safetensors 分片，峰值只需 1 个分片），
并把 fused experts（`mlp.experts.gate_up_proj` / `down_proj`，无 `.weight` 后缀）通过
`LLMTemplate.get("qwen3_5_moe")` 自带的 `f2f_weight_converters`（`SplitFusedExperts`）拆成 per-expert
`experts.{i}.{gate,up,down}_proj.weight` 后才可被量化。

驱动脚本：`quantize_ornith_int8.py`（`--coverage experts|experts_attn`）
耗时：v1 68.6 min（CPU 模式 + 并行 v2 抢占），v2 36.1 min。

## 3. 验证证据

**产物完整性**（`verify_ckpt.py`，v1）：`[verify] PASS`
```
[1] config: 架构字段保持；quant = int8 per-channel weight + dynamic per-token input
[2] index: 187124 tensors / 123 shards, missing files: 0, total 421.0 GB
[3] expert int8 tensors 92160/92160, scales 92160/92160
[4] per-channel int8 relative error over 1536 expert tensors: median 0.0039 p95 0.0039 max 0.0039
[6] missing tokenizer/processor assets: []
```
0.0039 = int8 对称逐通道量化的理论上界（0.5/127 of row amax）→ **量化数学正确**。

**v1 服务**（`EVIDENCE_real_int8.txt`）：
```
Using TRITON Int8 MoE backend out of potential backends: ['TRITON', 'HUMMING', 'CPU'].
Loading weights took 363.28 seconds
GPU KV cache size: 331,954 tokens
generated 256 tokens in 6.81s -> 37.57 tok/s (single request)
```
真实请求输出连贯（正确答出 "Aldebaran"、INT8 节省显存/带宽/算力三点分析）。

**v2 服务（aiter 路径）**（`EVIDENCE_v2_aiter.txt`）：
```
[gfx90a-patch] registered vllm.rocm_aiter_w8a8_gemm (upstream gate skipped it on CDNA2)
[gfx90a-patch] rocm_aiter_ops.is_linear_enabled restored on CDNA2
Selected AiterInt8ScaledMMLinearKernel for QuarkW8A8Int8          <-- 内核选择
[aiter] import [module_gemm_a8w8] under /jit/build/module_gemm_a8w8.so   <-- gfx90a JIT 产物被加载
[aiter] shape is M:164, N:4096, K:1024, q_dtype_w:torch.int8 ...        <-- 运行时真实调用(144 次 shape 记录)
generated 256 tokens in 7.44s -> 34.39 tok/s
```

**质量对比**（同一段 163 token 文本的 mean token NLL，`nll_probe.py`）：

| 型号 | mean NLL | perplexity | 单请求解码 |
|---|---|---|---|
| v1（仅 experts INT8） | **1.9687** | 7.16 | 37.57 tok/s |
| v2（experts + attn/GDN INT8，aiter） | **2.0239** | 7.57 | 34.39 tok/s |

v2 相对 v1 质量下降约 **+2.8% NLL**，属于把注意力/GDN 投影也量化后的合理代价。

## 4. 关键结论：gfx90a 上的 INT8 内核真相

1. **INT8 MoE 没有 AITER 后端**（与架构无关）：vLLM 0.29.x 的 int8 MoE oracle 只有
   `TRITON / HUMMING / CPU`，因此 512-expert INT8 走 **Triton W8A8 fused-MoE**（gfx90a 可用）。
   `--moe-backend aiter` 对 int8 会直接报错。
2. **INT8 Linear 的 AITER 内核在 gfx90a 上可编译且数值正确**：aiter 预编译 .so 只含
   gfx942/gfx950，首次调用会为 gfx90a 重新 JIT 编译 CK 内核（本次已编译完成，
   产物持久化在 `aiter_jit_build/module_gemm_a8w8.so`，187MB）；直接调用
   `aiter.gemm_a8w8` 实测 **max_rel_err 0.0019**。
3. **vLLM 默认在 MI250X 上不会用 aiter**：`is_aiter_found_and_supported()` 要求
   `get_cdna_version() > 2`（MI250X 是 CDNA2），导致 `rocm_aiter_w8a8_gemm` 自定义 op
   从未注册、`is_linear_enabled()` 返回 None。补丁 `aiter_patch/sitecustomize.py`
   只做两件事（注册该 op + 用 env 判定替换门控），其它 aiter 功能仍保持上游门控。
   用法：`-e PYTHONPATH=/patch -v <repo>/aiter_patch:/patch -e AITER_JIT_DIR=/jit/build
   -v <repo>/aiter_jit_build:/jit/build -e VLLM_ROCM_USE_AITER=1`。

## 5. 复现 / 运维要点

```bash
# 量化（CPU 即可，逐分片流式）
python3 quantize_ornith_int8.py --model <SRC> --out <DST> --device cpu [--coverage experts_attn]
# 校验
DST=<DST> python3 verify_ckpt.py
# 启动（v1）
./serve_int8.sh            # MAX_SEQS=16, MEM_UTIL=0.95, MAX_LEN=32768
# 启动（v2 + aiter int8 linear）
MODEL_DIR=<v2路径> NAME=ornith-int8-v2-aiter ... （见 EVIDENCE_v2_aiter.txt 的完整命令）
# 监控（失败关键字即退出，别只等 /health）
./watch_container.sh <容器名> 900 8100
# 验证
./smoke_serve.sh ; docker run --rm --network host -v $PWD:/work <image> \
  -c "python3 /work/nll_probe.py 8100 <served-name> <label>"
```

**显存/并发约束**（混合 GDN 架构特有）：权重 ~52.6GB/卡（v1）或 ~51.8GB/卡（v2），
`--gpu-memory-utilization` 需 ≥0.95 才有 KV 空间；45 层线性注意力的 Mamba state cache
与 KV 共享余量，因此 **`--max-num-seqs` 不能高**（默认 256 会报
`max_num_seqs exceeds available Mamba cache blocks`；本机使用 16）。

## 6. v3：INT4 (MXFP4 W4A16) + MTP 推测解码（2026-09-16 追加）

| 项目 | 值 |
|---|---|
| 产物 | `/mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-Quark-MXFP4-W4A16` |
| 大小 | **239.1 GB**（123 分片 / 187,124 张量；int8 版为 421 GB，**省 43%**） |
| 格式 | `dtype=fp4, qscheme=per_group, group_size=32, scale_format=e8m0`，权重 U8 打包 + e8m0 scale，**无激活量化（W4A16）** |
| 量化耗时 | **14.8 min**（GPU 模式；同配方 CPU 模式约 2 小时 → 差 8 倍） |
| 校验 | `verify_ckpt_mxfp4.py` PASS：专家张量 92160/92160 + scale 全覆盖，真实专家张量 normRMSE **0.112**、cosine **0.9937**，与独立 RTN 参考实现一致（0.1146）；nibble 顺序确认为 lo_first |
| 服务 | TP8 加载成功；`Using 'EMULATION' Mxfp4 MoE backend`；**KV cache 1,024,455 tokens**（int8 版 331,954） |
| MTP | `Resolved architecture: Qwen3_5MoeMTP` 生效；drafts 427 / accepted 341 → **接受率 79.9%** |
| 真实推理 | 正常且连贯（512 token 中文长回答，内容准确） |
| 速度 | **7.16 tok/s**（单请求，MTP 开）—— 比 int8 的 37.57 tok/s **慢 5 倍** |
| 质量 | mean token NLL **1.9868**（ppl 7.29）vs int8 1.9687 → **几乎无损（+0.9%）** |

**结论**：在 MI250X (CDNA2) 上 **INT4 是"省显存不省算力"**。gfx90a 有 INT8 矩阵核心但没有 FP4 硬件，
MXFP4 的 MoE 只能走 emulation（每次调用先反量化再 bf16 GEMM），且 MoE 只有 EMULATION 后端可选，
因此吞吐大幅下降；而 INT8 版本走 Triton int8 fused-MoE（原生 int8 指令）所以快。
质量上 MXFP4 出奇地好（NLL 与 int8 相差 <1%），MTP 接受率也很高（79.9%）。

**踩坑记录**：开启 MTP 后，草稿模型的专家层是**未量化 bf16**，vLLM 会自动为其选择 `ROCm AITER`
MoE 后端并报 `Unquantized MoE backend ROCm AITER does not support the deployment configuration`；
解决：`-e VLLM_ROCM_USE_AITER_MOE=0`（让未量化 MoE 走 Triton），已写入 `serve_v3_mtp.sh`。

---

## 7. v4：CT-Int4 W4A16（compressed-tensors）+ MTP —— 单流最好成绩（2026-09-17 追加）

口径：**单流 tps**（256 token greedy、MTP(1)、`max_model_len=262144`、无 YaRN、TP8×MI250X）。

| 配方 | 引擎/env | 单流 TPS | KV 池 | 接受率 |
|---|---|---|---|---|
| **int4 CT-W4A16 gs=128 + MTP(5) @256K** | **原生 vLLM 0.28.0**（8116 脚本，`SPEC=5`） | **68.83（5 次中位数）** | 1,555,665（5.93×） | 1915 drafts/898 accepted |
| int4 + MTP(6) | 原生 0.28.0 | 61.92（5 次中位数） | 1,512,249（5.77×） | 187/408 = 45.8% |
| int4 + MTP(4) | 原生 0.28.0 | 54.78（5 次中位数） | 1,599,602（6.10×） | 185/280 = 66.1% |
| int4 + MTP(3) | 原生 0.28.0（`SPEC=3`） | 54.10 | 1,653,036（6.31×） | 178/240 = 74.2% |
| int4 + MTP(2) | 原生 0.28.0 | 48.26 | 1,677,412（6.40×） | 154/202 = 76.2% |
| int4 + MTP(10) | 原生 0.28.0 | 46.56 | 1,465,930（5.59×） | 193/630 = 30.6% |
| **INT8-v2 + AITER Linear + MTP(1) @256K** | 原生 0.28.0（8115 脚本 + 旁挂补丁） | **43.35** | 534,131（2.04×） | 107/148 = 72.3% |
| int4 + MTP(1) @256K | 原生 0.28.0（8116 启动脚本） | 41.59 | 1,721,550（6.57×） | 114/142 = 80.3% |
| 同上 | nightly docker 0.29.1rc1 | 38.17 | 1,853,658（7.07×） | 88.9% |
| int4 + MTP(8) | 原生 0.28.0 | 37.82 | 1,476,727（5.63×） | 173/656 = 26.4% |
| INT8-v2 W8A8 + MTP(1) @256K（attn/GDN 也 int8，TRITON Linear） | 原生 0.28.0（8115 脚本） | 37.73 | 536,203（2.05×） | 118/137 = 86.1% |
| INT8-v2 @256K + ngram(5) | nightly | 19.31 | 276,736 | — |
| int4 @256K + ngram(5) | nightly | 17.78（NLL 最好 1.9558） | 1,732,860 | — |

**当前单流最好：int4 + MTP(5) = 68.83 t/s（5 次请求中位数）**，n=1 基线 40.66/41.59（两次独立测量一致）
⇒ **+69%**。n=5 同时是"步长最短（48.6 ms）+ tokens/步 最高（3.34）"，物理自洽。台账 **L9** 判死"加深 MTP"
（n=2 即峰）是 **llama.cpp/GLM-5.3-Flash**，本 engine/model 上**不成立**。

⚠️ **两个必须记住的测量事实**：
1. **臂内离散度 12–24%**（n=4 11.8% / n=5 16.7% / n=6 24.0%）⇒ 单次 256-token 请求的噪声是
   项目噪声门 1.036× 的 **5–20 倍**。**以后单流结论必须用同服多请求中位数**（`mtp_reps.sh` 口径）。
2. 单样本曾把 n=5 报成 64.95、n=6 报成 59.10；复测后中位数是 **68.83 / 61.92** ⇒ 单样本可差 ±7%。
   按步长分解（固定项 ≈27 ms/步 + 9.64 ms×tokens/步）n=5 单样本的 56.5 ms 步长本身也不自洽。

**步长分解（最小二乘，n=1 与 n=6 两点）**：步长 = **27.0 ms 固定开销 + 9.64 ms × （tokens/步）**。
校验：n=3 预测 57.8（实测 58.8 ✓）、n=4 预测 62.3（实测 62.9 ✓）。
⇒ **单流步长里 61% 是与 token 数无关的固定开销**（TP8 每步约 120 次 allreduce + 发射间隙 + 小 M 效率）
⇒ 这正是 MTP 有效的机理（摊薄固定项），也说明"换内核/调 tile"在单流上的天花板受同一约束。

KV 池代价小：n=5 时 1.556M tokens（−8.7% vs n=1），仍是 int8 臂（536k）的 2.9 倍。

### 7bis. 单流 TPS 的口径修正：**它强依赖文本可预测性**（2026-09-17 实测）

同一服务（SPEC=5、256K、greedy）下换 prompt，TPS 差 **56 → 88 t/s（1.57×）**：

| 负载类型（prompt） | 中位数 | 臂内离散 |
|---|---|---|
| C3 创作（低可预测，"写诗再翻译"） | **56.21** | 1.0% |
| A/B 解释型（"解释 int4 为何省带宽"） | 56.47（同 prompt）/ 69.88（加盐） | 13.1% / 7.3% |
| C1 数数（高可预测，"从一数到九十"） | **85.96** | 1.5% |
| C2 复述原文（极高可预测） | **87.54** | 1.6% |

**两条必须遵守的测量纪律**（修正既有做法）：
1. **首次请求必须丢弃**（预热伪影）：A 臂首次 49.06，其后 54.99–56.49（去掉后离散 2.7%）。
2. **固定 prompt 时计时噪声只有 1–1.6%**；**每次换 salt 会把中位数移动最多 24%**（内容变化改变 MTP 接受率）
   ⇒ 之前报的"臂内离散 12–24%"**几乎全是内容变化，不是计时噪声**；
   **A/B 要用固定 prompt ×5 才有足够信噪比**（加盐中位数只能定到 ±10%，检不出 5% 级效应）。
   这与项目噪声门 1.036× 的关系：**该门槛只在"固定 prompt"口径下成立**。

⇒ 因此本文所有单流数字都要带**负载标签**；"int4+MTP(5) = 68.83 t/s"是**加盐混合负载**的中位数，
开放分析型负载 ~56、结构化可预测负载 ~86。`mtp_reps.sh` 的历史数字属"加盐"口径（±10%）。

### 7ter. 固定开销消融（2026-09-17，**固定 prompt + 丢弃首次请求 ⇒ 离散 0.1%**）

口径：SPEC=5、256K、count prompt、5 次取中位数且**首次请求作为预热丢弃**。

| 臂 | 单流 TPS | vs 控制 | 判定 |
|---|---|---|---|
| 控制（当时默认） | 85.63（离散 **0.1%**） | — | — |
| **`--no-enable-prefix-caching`** | **88.82** | **+3.7%** | ✅ **已采纳进 8116 启动脚本** |
| `--compilation-config {"cudagraph_mode":"PIECEWISE"}` | 78.17（离散 0.1%） | **−8.7%** | ❌ 默认 `FULL_AND_PIECEWISE` 明显更优 |
| `--max-num-seqs 1` | 81.89 | **−4.4%** | ❌ 保持 16 |
| `--compilation-config {"pass_config":{"fuse_allreduce_rms":true}}` | — | — | ❌ **本 build 无该 pass**：`name 'AllReduceFusionPass' is not defined` |
| `NCCL_MIN/MAX_NCHANNELS=1`（batch 3 精确重测） | **78.68**（离散 2.5%） | **−12.1%** | ❌ 别动 RCCL 默认通道数 |
| `disable_padded_drafter_batch:true`（batch 3） | 89.05 | −0.5% | ❌ |
| `use_local_argmax_reduction:true`（batch 3） | 89.78（区间 [89.75,89.78] 与控制不重叠） | +0.3% | ⚠️ 统计真实但**在噪声门 1.036× 内**，不采纳 |
| `--async-scheduling`（batch 4） | 89.26（离散 0.1%） | −0.5% | ❌ |
| `--block-size 128` → **实际被改 640**（batch 4） | 82.85 | −7.6% | ❌ |
| `--block-size 256` → **实际被改 768**（batch 4） | 88.52 | −1.3% | ❌ |

**batch 4 的重要副产品**：本模型是**混合结构**（45 层 GDN + 15 层 full-attn），
**`--block-size` 不是独立旋钮**——mamba 页对齐逻辑会把它重新缩放（128→640、256→768，
日志 `Setting attention block size to N tokens` 可证），而自动选出的 **544 反而最优**
⇒ "调 block size" 这条路是**被封的，不是没试**。

#### batch 4bis：splitKV 在 MTP 下为何永不接管（机制确认，2026-09-17）

启动脚本一直写着 `splitKV=1(SPEC>0 时不接管)`——本轮给出了**机制证据**，从假设升级为结论：

- 补丁**确实在跑**：日志有 `rocm_splitkv_pa.py:509] MI250X split-KV: scratch 基座扩…`（它在**图捕获期**扩了 scratch）
- 但捕获时 **MTP(5) 每步 query 长度 = 6 > 1**，判定门槛要求 `max_query_len==1` ⇒ **拒绝接管**
- ⇒ 被捕获进图的是 **Triton 回退路径**，日志 `Cannot use ROCm custom paged attention kernel, falling back to Triton implementation.`

⚠️ **诊断工具的一个坑**（记录以免重踩）：补丁的统计文件由 `VLLM_ROCM_SPLITKV_PA_STATS_EVERY` 控制落盘，
**默认 0 = 永不落盘**；而判定在**图捕获期只发生 ~10–30 次**，所以 `EVERY=400` 也永远不落
⇒ 想拿 `reject_by_reason` 必须把 EVERY 设到个位数（且要意识到"统一 decode 步在图里不跑 Python"，
统计天然只能反映捕获期与预填充期）。

**`--no-enable-prefix-caching` 的机制**（这是它有效的原因，不是玄学）：
`config.py:602` 只要 `enable_prefix_caching` 为真，就把 `mamba_cache_mode` 从 `none` **强改为 `align`**；
`align` 会带来每步的页对齐内核（`precopy_mamba_align_fused_kernel` / `postprocess_mamba_fused_kernel`）、
3 个 padding 层、以及 6.67% 的 KV 浪费。**而 MTP 下前缀缓存永不命中**（本会话实测 `hits 0`）
⇒ 关掉是纯赚：**+3.7% TPS 且 KV 池 1,529,086 → 1,564,999（+2.3%）**，日志里 `Mamba cache mode is set to 'align'` 消失。
⚠️ **只对本臂（MTP）成立**；ngram 臂靠前缀缓存复用长文档，那边必须保持开启。

**负面结论同样有价值**：`FULL_AND_PIECEWISE`（默认）比 `PIECEWISE` 高 **8.7%** ⇒ **不要把 attention 移出整图捕获**；
`max-num-seqs=1` 反而慢 4.4%（尽管 KV 池更大 1,758,397）⇒ 单流不需要收窄 seqs。

**方法论三条**（对以后所有单流 A/B）：
1. **固定 prompt + 丢弃首次请求 ⇒ 离散 0.1%**（greedy 确定性 ⇒ 输出、接受率、时延都可复现）；
2. 加盐 prompt 的中位数只能定到 **±10%**，检不出 5% 级效应；
3. **首次请求是预热伪影**（A 臂首次 49.06 vs 其后 54.99–56.49；本批还见到 warmup 比稳态低 15–20%）。

### 7quater. 成品配方（2026-09-17 定稿，三种负载口径）

配置 = `int4 CT-W4A16 gs=128` + 原生 vLLM 0.28.0 + TP8 + `SPEC=5` + **`--no-enable-prefix-caching`** + 256K。

| 负载 | 单流 TPS 中位数 | 离散 | 采纳该开关前 | 净增益 |
|---|---|---|---|---|
| 结构化/可预测（count） | **89.78** | 0.2% | 85.63 | **+4.8%** |
| 开放分析（explain，同 prompt ×5） | **63.38** | 4.9% | 56.48 | **+12.2%** |
| 混合（加盐 ×5） | **64.66** | 17.4% | ~62.53 | +3.4% |

**低接受率负载受益更大**（+12.2% vs +4.8%）：`align` 机制是**每步固定成本**，
在高接受率（tokens/步 更多）的负载上被摊薄得更多 ⇒ 与 §7ter 的机制解释一致。

### 7septies. GEMV 式 decode MoE 内核：原型已跑通、快 2×，**数值待修**（2026-09-17，进行中）

**已完成**
1. **现役内核耗时测量**（`bench_wna16_moe.py`，直调 vLLM 的 `invoke_fused_moe_wna16_triton_kernel`
   + 合成张量，本模型形状/TP8 分片）：见 §7sexies —— MoE 占步长 **46%**，只跑到 HBM 峰值 **~12%**。
2. **GEMV 式原型内核**（`gemv_moe.py`）：不做排序、不 padding M，
   **一个 program 一个 (token, expert) 对 × N tile**，真 GEMV（不用 `tl.dot`），
   输出每对的部分和 `[M*topk, N]`（与 vLLM 的 moe_sum 阶段解耦，便于同口径比对）。
   计时（同 harness、同 shape，gemm1 现役 = 213.9 µs）：

   | 配置 | 我的内核 | vs 现役 |
   |---|---|---|
   | BLOCK_N=64, BLOCK_K=128, w4 | 130.8 µs | **+64%** |
   | **BLOCK_N=128, BLOCK_K=128, w4** | **106.1 µs** | **+102%** |
   | BLOCK_N=256, BLOCK_K=128, w8 | 108.2 µs | +98% |
   | BLOCK_N=64, BLOCK_K=64, w4 | 140.4 µs | +52% |

   ⇒ **gemm1 快 2×**（且完全未调优）。若落地：每层省 ~106 µs ⇒ ×60 层 = **−6.4 ms/步 ⇒ 单流约 +20%**。

**尚未完成（唯一卡点）**：**数值仍不一致**（与自建参考差 138%、与现役内核对拍差 138.7%）。
根因是我还没把 vLLM **TRITON 后端**的 B 运行时布局钉死：
- 已确认：内核按 `(offs_k//2)*stride_bk` 寻址、`b_shifter=(offs_k%2)*4`、**`b_zp_num=8`**
  ⇒ **每元素 2 个 nibble（低 nibble = 偶 k）+ offset-binary（真值 = nibble − 8）**；原型已按此实现。
- 已确认：`apply` 直传 `layer.w13_weight`，但加载后有 `convert_to_wna16_moe_kernel_format(...)`
  （定义在 `fused_moe/oracle/int_wna16.py:1407`）——**该函数对 TRITON 后端做了什么转换，是最后一块拼图**
  （我读到的都是 Marlin/Humming 分支，TRITON 分支未读完）。
- 探针工具已就绪：`pin_wna16_layout.py`（用 M=1/TOPK=1 把 MoE 退化成单次 GEMV，逐假设对拍：
  它已把"2 nibble/元素 + offset-binary"证到 **0.20% 误差**）、`cmp_mine_vs_incumbent.py`（我的 vs 现役，同输入）。

**✅ 已完成（2026-09-17 收尾）**：三步都做完了。
- **布局钉死**：`oracle/int_wna16.py:1630` 的 TRITON 分支（compressed-tensors 走 `QuantizationArgs`）——
  `w13.transpose(1,2).contiguous().view(torch.uint8)` ⇒ 内核收到的是 **`B: uint8 [E, N, K//2]`**
  （**每字节 2 个 int4**，低 nibble = 偶 k，真值 = nibble − 8），`scale: [E, N, K//group]`。
  （我早先按"int32 每元素 8 nibble"写的版本因此偏慢/算错；**旧数字 213.9 µs 是喂错布局的假象**。）
- **内核已按真布局改写并数值通过**（`gemv_moe.py`、`gemv_moe_ab.py`，真实随机路由）：
  与 **现役内核直接对拍**：gemm1 **2.58%**、gemm2 **0.14%**（fp32 累加 vs 现役 bf16 部分和，属预期差异）。

| M=6、真实路由 | 现役 | 我的 GEMV（BN=128, BK=128, w4） | 提升 |
|---|---|---|---|
| gemm1（59 个不同专家） | 185.1 µs | **114.4 µs** | **+62%** |
| gemm2（57 个不同专家） | 60.9 µs | **37.7 µs** | **+62%** |
| 每层 | 246.0 µs | 152.1 µs | 省 **93.9 µs** |

⇒ **×60 层 = 省 5.63 ms/步** ⇒ 步长 37 → 31.4 ms ⇒ **单流预计 +18%**（89.78 → ~106 t/s，count 口径）。

**集成尝试（2026-09-17，未成功，状态已定位）**：写了 `moe_gemv_patch/{mi250_moe_gemv.py,sitecustomize.py}`
（flag `MI250_MOE_GEMV`，懒加载 monkeypatch `triton_moe.invoke_fused_moe_wna16_triton_kernel`，异常兜底回上游；
**默认关闭，不影响正常起服**）。实测：
- ✅ 补丁装上并**正确接管 decode 调用**（日志 `patched …`、`takeover` 增长、形状 `M=…/K=4096/N=256`）；
- ✅ 通过 DUMP 发现并修掉一个真 bug：**prefill 调用（M=2048、pairs=20480、BM=64）绝不能接管**（否则输出坏 + 反向变慢）；
  已加 decode-only gate（`M<=64 且 pairs<=256`）。
- ❌ **端到端仍不正确**：NLL **12.92**（基线 ~1.97）、TPS 88.2（基线 89.6）⇒ 无收益。
  自检（同一次真实调用上跑上游并对拍）给出：`GATE: M=24 pairs=240 num_valid=240 distinct_experts=10`
  （期望 ~57）与 `SELFCHECK: 993%~1532%` ⇒ **集成层的"逆置换"在真实 padding 路由下赋错了专家**。
  （自检误差也混入已知因素：上游对 padding 行写 0、我的内核会真算。）

**集成尝试第二轮（stash topk_ids，2026-09-17）**：改 hook `TritonWNA16Experts.apply` 把 `topk_ids` 暂存给内核
（彻底删掉逆置换）——补丁装上了（日志 `patched TritonWNA16Experts.apply (stash topk_ids)`），
但**端到端仍不正确**：NLL **12.9161**（三次运行完全一致）、TPS **90.22**（基线 89.6，+0.7%，在噪声内）。

⚠️ **一个方法论教训**：我用的"自检"（同一次真实调用上跑上游并对拍）**信号不干净**——
它比较全部 240 对，其中 **180 对是 padding 行**（上游对 padding 写 0、我的内核会真算；padding 的 `topk_ids`
还可能是 -1 → 我内核按负专家号寻址得到垃圾）。1000–4000% 的差值因此**不能作为"真实对也错"的证据**，
我据此又追了一轮，属判断失误。

### ✅ 第三轮：**结论反转——内核正确、集成可用、收益 +0.5%**（2026-09-17 收尾）

**三个此前的"失败信号"全部被证明是假警报**：

| 此前的判断 | 真相 | 证据 |
|---|---|---|
| 自检 `我的 vs 上游 = 993%~4000%` ⇒ 集成错 | **自检自身写错**：它插在内核 launch **之前**，比较的是**上一层遗留的脏 C 缓冲** | 把捕获的真实张量拿到离线重跑我的内核 → **0.14%**，逐对比值 1.000 |
| NLL 12.9161 ⇒ 输出坏 | **与本补丁无关**：补丁**关掉**也是 12.9161 | `MI250_MOE_GEMV=0` 同配方实测 NLL=12.9161（同一数值） |
| TPS 无变化 ⇒ 无收益 | 实为 **+0.5%**（89.51 → 90.0/90.22） | 关：89.51（离散 0.1%）；开：90.00 / 90.22（两次运行一致） |

**⇒ 现状（修正后）**：
- 内核 **数值正确**，三方验证：合成 harness **0.15%**、**真实捕获张量 0.14%**、上游 **0.16%**（对同一独立参考）。
- 集成 **可用**：只接管 decode gemm1、prefill 安全回退、异常兜底回上游、flag 默认关。
- **真实收益 +0.5%**（不是预估的 +11%）。

**为什么只有 +0.5%**：真实 decode 路由**只有 10 个不同专家**（我 harness 用随机路由是 59 个）⇒
现役内核在真实负载下远快于我 harness 里测的 185 µs/层，**MoE 真实占比与"+62%"都要按真实路由重测**。
我先前"MoE 占步长 40%"的结论同样建立在这个未经真实路由验证的 harness 上，**需重新定价**。

### 🔧 NLL 探针污染：已定位并修复（2026-09-17）

**根因**：本机 vLLM(0.28) 在 **SPEC=5（MTP 深度 5）** 下，`prompt_logprobs=0` 与 `echo=True+logprobs=1`
**两条独立路径都**返回近似均匀随机的 logprob ⇒ NLL ≈ ln(V) ≈ 12.9。**不是模型坏、不是探针写错**
（token 对齐已核对正确：`'The',' AMD',' Inst','inct',' MI','2','5','0','X'…`）。

**隔离实验（同一 launcher，只改一个变量）**：

| 配置 | NLL | 判定 |
|---|---|---|
| SPEC=0（无投机） | **1.9638** | ✅ |
| SPEC=3 | **1.9654** | ✅ |
| SPEC=5 | **12.9161 / 12.9204 / 12.9113** | ❌ 三种变体都坏 |
| SPEC=5 + 前缀缓存**开** | 12.9204 | ❌ ⇒ **与 `--no-enable-prefix-caching` 无关** |
| SPEC=5 + `--max-num-batched-tokens 4096` | 12.9113 | ❌ ⇒ 与投机 token 预算无关 |

⇒ **触发条件就是 SPEC=5 本身**（SPEC 0/3 干净；4/6/8/10 未测）。**速度与接受率指标不受影响**
（MTP 接受率 88.9% 正常 ⇒ draft/target 一致 ⇒ 生成路径本身没问题，坏的只是 logprob 上报）。

**修复（已落地 `nll_probe.py`）**：内置闸门 —— `NLL > 8`（≈均匀分布量级）时**拒绝采信并 exit 2**，
并提示改在 SPEC=0/3 下测质量。**质量测量规程**：**质量一律在 SPEC≤3 下测**（速度可继续用 SPEC=5）。

**污染清单（本会话全部 NLL 测量）**：

| 文件 | 值 | 配方 | 是否被污染 |
|---|---|---|---|
| `NLL_v1-experts` | 1.9687 | int8 v1（experts-only） | ✅ 干净 |
| `NLL_v2-aiter-experts_attn` | 2.0239 | int8 v2 + aiter | ✅ 干净 |
| `NLL_v3-mxfp4-mtp` | 1.9868 | mxfp4 + MTP | ✅ 干净 |
| `NLL_U1-int4-256k-ngram5` | 1.9558 | int4 + ngram(5) | ✅ 干净 |
| `NLL_int8-256k` | 2.0645 | int8 @256K | ✅ 干净 |
| `NLL_int8-aiter-mtp3` | 1.9749 | int8 + aiter + **MTP(3)** | ✅ 干净（SPEC=3 今日复核 1.9654） |
| **`NLL_moe-gemv-on` / `NLL_moe-gemv-OFF`** | **12.9161** | int4 + **SPEC=5** | ❌ **污染** |

**⇒ 被推翻的结论**：**只有一个** —— 我在 §7septies 第三轮里写过的"我的补丁把输出弄坏了（NLL 12.9）"，
**该结论作废**（已在该节更正为假警报：补丁关掉也是 12.9161）。**既有项目的质量结论无一条被推翻**
（它们的 SPEC 都 ≤3）。

**⇒ 受影响但"未验证"而非"被推翻"的**：**出厂配方（SPEC=5）本身没有有效的质量测量**。
代理口径：SPEC=3 + 同配方 = **1.9654** ✅（说明权重质量正常）；SPEC=5 臂的 logprob 上报仍需
vLLM 侧修复或用 SPEC≤3 代理。**此前那条"aiter 臂接受率掉 13.8 点但不代表质量下降（NLL 1.9749）"
依然成立**（MTP(3)，干净）。

### ❌ 第四轮：**真实形状下我的内核反而慢 26% ⇒ 结论：不启用**（2026-09-17 终判）

用捕获的真实张量按**真实 decode 形状**定价（`price_real.py`，REP 放大到 M=24）：

| 形状（真实路由，10 个不同专家） | 现役 gemm1 | 我的 GEMV | 差 |
|---|---|---|---|
| M=8 → 80 对 | 191.8 µs | **111.7 µs** | **+72%** |
| **M=24 → 240 对（vLLM 实际调用的形状）** | **194.1 µs** | 261.4 µs | **−26%** |

**⇒ 在 vLLM 实际调用的形状上我的内核更慢**，端到端 +0.5% 正是"部分步小赢 + 部分步小输"的净值。

**根因（结构性，不是实现 bug）**：真实 decode 的 M=24 是 vLLM **按 cudagraph 捕获尺寸 padding** 出来的；
我本想消除的 padding 是**上游强加的**。在 240 行规模下，现役的 `tl.dot` 分块已足够高效，
而我的真 GEMV（无 `tl.dot`、纯标量 FMA）只在 **≲80 对**时才占优 —— **恰好落在 vLLM 不会调用它的区间**。

**⇒ 决定：不启用该补丁**（`MI250_MOE_GEMV` 默认 0，保持不变）。这条 MoE 内核线**到此收口**：
- 内核本身**正确**（三方验证 0.14~0.16%），集成**可用且安全**，但**在真实调用形状上无收益**；
- 现役 MoE 在真实形状下约 **194 µs/层 → ~11.6 ms/步 → 约占步长 31%**（比 §7sexies 的 40% 低，因为
  之前用了非代表性的随机路由），但**它在自己的形状上已经高效**，要赢必须击败 `tl.dot`，难度完全不同量级；
- 若真要再动 MoE，唯一有希望的方向是**减少上游 padding 本身**（需要改 vLLM 的 cudagraph 批次语义，非内核问题）。

**下一轮应当用干净的判据**（按性价比排序）：
1. **自检只比真实对**：在 `apply` 里同时暂存 `hidden_states.size(0)`（**真实 token 数**，不是 padding 后的），
   自检只比较 `[:real_pairs]`；再把 padding 的 `topk_ids` 钳到 [0,E) 避免负号寻址。
2. **端到端单步 A/B**：`MI250_MOE_GEMV=0` 与 `=1` 各生成 32 token，逐 token 比对 —— 若第 1 步就分叉，
   说明 decode MoE 确实错；若前若干步一致而后续分叉，则是别处（如 padding 行影响了后续状态）。
3. 若 1、2 都指向"真实对正确"，则 NLL 的崩坏另有原因（例如 padding 行的垃圾值被下游用到）。

**下一步的两个候选（按把握排序）**：
1. **改挂钩点**：不hook `invoke_*`（那里只有 sorted/eids，需要逆置换），改为 hook `TritonWNA16Experts.apply`
   （`experts/triton_moe.py`）——**那里 `topk_ids` 是现成的**，我的内核直接吃它，**完全不需要逆置换**，
   这一整类 bug 消失（代价是要在原函数内插入调用，改动更大）。
2. **把真实数组 dump 出来对账**：`sorted_token_ids[:80]` + `expert_ids[:20]` + padding 常量，
   与"期望映射"逐条比对，定位逆置换的语义差（例如 padding 常量与 `num_valid_tokens` 不等、
   或 padding 条目与真实 pair 下标重叠导致后写覆盖）。

**剩下最后一步（未做）**：按项目惯例接进 vLLM——
`patches/gfx90a/kernels/mi250_kernels.py` 已有 `MI250_GATE_GEMV/TOPK/MOE_ACT/MOE_SUM` 四个 flag+monkeypatch
先例，新增一个 `MI250_MOE_GEMV` 接管 WNA16 的 MoE（gemm1/gemm2 都走我的内核），
再端到端测 TPS 与 NLL。**注意**：需处理 MoE 的 padding/sorting 语义差异（我的内核直接吃 `topk_ids`，
不需要 `sorted_token_ids`/`expert_ids`），以及 `moe_sum` 阶段的接口对齐。

### 7sexies. **MoE 的真实耗时终于测到了：占单流步长 46%**（2026-09-17，微基准）

整场 session 都缺这个数（profiler 不可用）。绕过办法：**直接调 vLLM 自己的
`invoke_fused_moe_wna16_triton_kernel`**，用合成张量按本模型形状/TP8 分片打点
（脚本 `bench_wna16_moe.py`）：E=512、topk=10、hidden=4096、moe_inter=1024、gs=128；
gemm1 `A[M,4096]×B[512,256,512]i32`、gemm2 `A[M*10,128]×B[512,4096,16]i32`（每 rank 分片）。

| 每步 token | gemm1 | gemm2 | 每层 | ×60 层 |
|---|---|---|---|---|
| 1 | 95.8 µs | 46.7 µs | 142.5 µs | **8.55 ms/步** |
| **6（MTP(5) 实测值）** | 213.9 µs | 66.9 µs | **280.8 µs** | **16.85 ms/步** |

**实测步长 ≈37 ms（89.78 t/s @ 3.34 token/步）⇒ MoE 占 45.7%（该值由错误布局下的 280.8 µs 得出，偏高；
用真实布局 + 随机路由修正后是 246.0 µs ⇒ 14.76 ms/步 ⇒ 约 40%，结论不变）。**
这**独立验证**了 §7quinquies 从 tile 敏感度反推的"30–50%"，并**彻底推翻**了我早先按权重带宽估的 ~6%。

**带宽下界**：每 rank 每 expert 0.75 MB × ~57 活跃专家 = 43 MB/层 → 2.5 GB/步 → **≈2.0 ms/步**
⇒ **现役内核只跑到 HBM 峰值的 ~12%**（M=1 时更差：只动 5 MB 却花 95.8 µs gemm1，比下界慢约 20×）。

**机制（可执行）**：M=1~6 时 `sorted_ids≈160`（10 个活跃专家 × BM=16 padding），
grid = (160/16) × (256/64) = **40 个 block**，在 104 CU 上**严重欠占用**，且每 block 串行 128 次 K 迭代
⇒ **occupancy/延迟受限，既不是带宽受限也不是算力受限**。这正是"GEMV 式、按 (token,expert) 并行"的靶子。

**⇒ GEMV 方案的奖金额**：把 16.85 ms 压到接近 2–5 ms ⇒ 步长 37 → ~25 ms ⇒ **单流 +40~50%**，
量级大于本会话此前所有 config 层优化的总和。

### 7quinquies. MoE 调优表（davetha 路线）实测：**默认已是最优，但暴露了 MoE 的真实占比**（2026-09-17）

机制：vLLM 自带 **331** 张 Triton MoE 调优表，**本机设备名 `AMD_Instinct_MI250X_MI250` 一张都没有**
（日志实锤 `Config file not found at .../E=512,N=128,device_name=AMD_Instinct_MI250X_MI250,dtype=int4_w4a16.json`），
我们全程跑 vLLM 的默认启发式（`BM=16, BN=64, BK=32, GROUP_SIZE_M=1`）。
用 **`VLLM_TUNED_CONFIG_FOLDER`** 外挂表（**不动 site-packages**）做单变量 A/B，
固定 count prompt ×5、丢弃首次请求（精度 0.1%）：

| tiles (BM/BN/BK/G/warps/stages) | 单流 TPS | vs 默认 |
|---|---|---|
| **16/64/32/1/4/2（vLLM 默认）** | **89.81 / 89.21**（两臂复测） | — |
| 16/64/16 | 84.92 | −5.5% |
| 16/64/64 | 81.84 | −8.9% |
| 16/32/64 | 86.22 | −4.0% |
| 16/32/32 (warps 2) | 87.04 | −3.1% |
| 16/128/64 | 80.77 | −10.1% |
| 16/256/32 (warps 8) | 68.89 | −23.3% |
| 32/128/64 | 55.59 | −38.1% |

**结论 1（否决该杠杆）**：默认 tile 在 (BM, BN, BK) 三个维度上**都是局部最优**，8 个方向全部更差
⇒ **为单流 decode 手写 MI250X 调优表没有收益空间**（davetha 的 +18% 是**并发聚合**口径，
大 M 形状未测，不是同一回事）。

**结论 2（更重要，且修正了我此前的估算）**：这张表的**敏感度**给出了 MoE 占比的下界证据——
- `BN 64→128`（N 工作量 ×2）⇒ 步长 **+11.2%**（+4.2 ms/步）
- `BM 16→32 且 BN 64→128`（M×N 工作量 ×4）⇒ 步长 **+61.5%**（36.7 → 59.3 ms/步）

⇒ **MoE 约占单流步长的 30–50%**，**不是我先前按"权重带宽下界"估的 ~6%**（那个估算只算了字节，
没算 padding 与解包计算）。机制：**BM=16 是内核下限**，而每步有 ~60 个不同活跃专家、
每个只摊到 ~1 个真实 token ⇒ **约 16× 的 padding 浪费**，属该内核的**结构性问题，调表治不了**。

**对既有结论的连带影响**：
1. 我在 `docs/MI250X-AITER-INT4-内核复核-2026-09-17.md` §2 里"内核天花板 ~2%"的说法**再次被削弱**——
   那是"权重字节"的下界；**MoE 的真实成本主要是 padding/解包，不是字节**。
2. 那"27–40 ms 固定开销"的成因也要改写：其中很大一块**不是**发射/allreduce，而是**MoE 的 padding 计算**。
3. ⇒ MoE 侧的内核工程（路径 A/B，或 GEMV 式 MoE）**收益上限比我先前定价的高得多**（但仍属大工程）。

**本会话该臂的完整增益**：nightly docker + MTP(1) 起点 **38.17 t/s** →
原生 env + MTP(5) + 去 align **89.78/63.38**（可预测/开放）⇒ **+66% ~ +135%**。
其中归因：环境切换 38.17→41.59（+8.9%）、MTP 深度 41.59→68.83（+65%，加盐口径）、
去 prefix-cache 再 +3.7~12%。

**aiter int8 Linear 匹配口径 A/B（单变量）：37.73（Triton）→ 43.35（aiter）= +14.9%，步长 −19.5%。**
注意 aiter 臂 **MTP 接受率反而低 13.8 点**（两套 int8 GEMM 数值不同 ⇒ draft/target 一致率变化）
⇒ 在更差接受率下仍快 15%；**但因此必须复核输出质量（NLL）**，不能只看速度。

**匹配对照（同 env / 同 256K / 同 MTP(1) / 同启动模板 / 同 TP8）：int4 比 int8 快 10.2%。**
但拆步长可知差值来自**量化覆盖范围**而非权重字节：int4 臂把 attn/GDN 留在 bf16，
每步少 6.0 ms（权重读取时间只差 0.85 ms）。推导见
`docs/MI250X-AITER-INT4-内核复核-2026-09-17.md` §3（该节同时更正了初稿里一组跨 maxlen 的混淆对照）。

**运维坑（三个都会白等几分钟才崩）**

1. **checkpoint 权限**：int4 checkpoint 由 root 容器生成，123 个权重是 `-rw------- root:root`；
   主机 qiba 身份**原生**起服必然 `PermissionError`（docker 里 root 读得动 ⇒ 只在原生路径暴露，
   且在加载 200 s 后才崩）。修 `sudo chown -R $USER:$USER <ckpt>`；8116 启动脚本已加
   fail-closed 权重可读性闸门（`权重不可读` 直接退出）。
2. **`--load-format fastsafetensors` 只在本模型 int4 上安全**：int8 权重 51.75 GiB/die
   （util 0.96，可用 61.4 GiB），该 loader 批量搬 shard 时一次申请 **12.29 GiB** 而只剩 8.88 GiB
   ⇒ `Consumer error: CUDA out of memory` + `gpu_model_runner.py:5513 Failed to load model`。
   int4 只有 28.4 GiB/die，故安全（加载 142.7 s vs 常规 286 s）。
3. **torch profiler 在 0.28.0 上挂不上**：`/start_profile` 需 `--profiler-config`（`VLLM_TORCH_PROFILER_DIR`
   无效），但一挂上 **EngineCore 在加载完成后直接死**（无 OOM、无 Python 异常，只剩 shutdown 噪声）
   ⇒ 拿不到 kernel 级分解。

**aiter 的新事实**：`module_gemm_a8w8` 的 gfx90a JIT 是一次性成本，**但发生在引擎初始化路径内**，
8 个 rank 串行等 `aiter/jit/build/lock_module_gemm_a8w8` ⇒ 首次起服要 20–40 min，
健康轮询/启动超时会被打爆（实测 1000 s 超时后 EngineCore 仍在等 shm broadcast）。
**结论：先单独预热编译，再起服。**

**不要做**：为单流自研 aiter/CK int4 内核 —— 台账 **L25/L26 `rejected`**
（`bench/ledger.py gate "AITER int4 内核"` → exit 1）。替代杠杆见该文档 §6（步数 > 发射/同步 > 覆盖范围）。

### 7octies. gfx90a 版 a8w8 调优表（2026-09-18，为消 432 行启动日志；**性能恒等**）

**产物位置（唯一权威副本）**：
`/home/qiba/ai/patches/gfx90a/aiter_a8w8_tuned_gemm_gfx90a.csv`
sha256 `f6a703059003ed8ed4d64aef9f72a74b1cef4ab7093d8efc4697d48356f77854`，634 行
= 官方表 579 行原样 + gfx90a 54 行（27 个 cudagraph 尺寸 × 2 个 shape）。
本目录下的同名文件**是符号链接**（不是副本）：同 `/home/qiba/ai` 树是 launcher 依赖所在，
两份硬拷贝必然会分叉，而符号链接不可能过期。

**为什么放 `ai/patches/gfx90a/` 而不是本工作区**：它是**运行期依赖**——
8117 启动脚本第 192 行 `export AITER_CONFIG_GEMM_A8W8=$A8W8_GFX90A_CSV`，与
`aiter_int8_patch/`、`kernels/mi250_kernels.py` 等 gfx90a 补丁同居一处。工作区保留的是
**可复现性资产**：生成器/测量器（`aiter_a8w8_tune.py`、`aiter_a8w8_ingraph_shapes.py`、
`aiter_a8w8_ingraph.py`）。

**注入方式**：`AITER_CONFIG_GEMM_A8W8=<本表>`。⚠️ 必须给**超集**：aiter 只在 env 含多条路径
（`:` 分隔）时才合并（`jit/core.py:216`），单条路径会**整体替换**官方表。
验证：同一批 shape **新表 54/54 命中、官方表 0/54**；注入后新进程不再打印
`not found tuned config`。关闭：`AITER_CONFIG_GEMM_A8W8= bash <launcher>`。

**为什么不可能提速（三层查证）**：① 这条路径是 `gemm_a8w8_CK`（`vllm/_aiter_ops.py:648`），
表里**只取 `splitK`**（`ops/gemm_op_a8w8.py:660-664`，`kernelId/kernelName` 只有 ASM 路径用）；
② gfx90a 上 `splitK=1..4` 全部 `RuntimeError: This GEMM is not supported!` ⇒ 唯一合法值 0，
即现有默认；③ 整块的真实成本（图内 replay 口径）**1.06 ms/步 = 步时的 2.6–2.7%**
（单 shape M=6：`in_proj_qkvz` 16.05 µs@653GB/s、`out_proj` 6.02 µs@697GB/s；纯访存下限 0.41 ms/步）
⇒ 换完美内核最多 +1.5%。所以本表的收益只有两项：消掉 432 行 INFO + 固化设备实测事实。

**⚠️ 口径陷阱（我先前错过一次，已自纠）**：那 432 行每 (M,shape) 只出现一次，
**不是**因为"capture 后走图"，而是 `get_GEMM_config_with_quant_type` 带
`@functools.lru_cache(maxsize=1024)`（`ops/gemm_op_a8w8.py:489`）
⇒ **不能用日志行数推断每步 GEMM 调用次数**。

**另一个更有价值的数**：**eager** 下单次调用 ~46 µs（其中 ~40 µs 是 host 开销）
⇒ 90 次/步 = 4.17 ms/步（步时的 10.8%），生产靠 cudagraph 才降到 1.06 ms/步
⇒ 这是"本臂绝不加 `--enforce-eager`"的又一条硬证据。

**重新生成**（换 aiter 版本或换模型后）：
```bash
~ROCm.AI/quark-int8$ /home/qiba/ai/envs/vllm_0.28.0_rocm72/bin/python aiter_a8w8_ingraph_shapes.py
# 会按图内口径逐 shape 重测，并合成超集表；末尾自带命中/对照自检（应 54/54 与 0/54）
```

### 7novies. MTP 与"前缀缓存失效"的根因追查（2026-09-18）——旧结论被更正

**结论**：在 **native vLLM 0.28.0 + 256K/seqs16** 上，**开 MTP 并不会让前缀缓存失效**。
实测（`--enable-prefix-caching` + SPEC=5，prompt≈2825 tok）：
`prefix_cache_queries +5616 / hits +2176`，而 **2176 = (2825//544 − 1) × 544**
⇒ 命中正好是"满块数 − EAGLE 丢的那一块"，是**正常的 EAGLE/MTP 代价**；
暖请求 TTFT **2.96 s → 0.49 s（6.0×）**。

**代价才是真问题**：`--enable-prefix-caching` 会把混合 GDN 模型强推进 mamba `align` 模式
（`model_executor/models/config.py:602-629`），税很重 —— 同一 prompt 下 decode
**107.80 → 82.20 t/s（−23.7%，n=3 spread 2.6%）**、冷 TTFT 0.91 → 2.96 s（n=1 待重复）。
（该臂 `/metrics` 步时分解内部不自洽：steps+accepted ≠ 生成 token 数 ⇒ 不采信其 step ms/accept%。）

**机制三层（file:line）**：
1. 配置层：开前缀缓存 ⇒ `mamba_cache_mode` none→'align'、`mamba_block_size` 由 `max_model_len` 改为 `block_size`；
   align 的语义是"只在某个 scheduler step 的末 token 且位置为 block_size 整数倍时才留状态快照"（`config/cache.py:141-148`）。
2. 模式层：高效档 `'all'`（每块边界都留快照）**Qwen3.5 直接 NotImplemented**（`models/qwen3_5.py:320-326`）。
3. 投机层：`use_eagle()` 对 mtp 为真、对 ngram 为假（`config/speculative.py:1477-1481`）；防"草稿污染末块"要对 KV 组做
   末块 drop，而 **Qwen3.5 没有任何组被标 `is_eagle_group`**（该标记只在 DeepSeek-V4 特例置位）⇒ 兜底把**所有组**都标上
   （`v1/core/kv_cache_coordinator.py:107-113`），mamba 组又被排除在该 drop 的 margin 之外（同文件 814-819）。

**为什么别的模型开 MTP 不坏**：纯注意力模型没有 mamba 组 ⇒ 既没有 align/状态快照约束，也不存在"递归状态无法按 token 回滚"，
EAGLE 丢末块对 append-only KV 正确且只损失 1 块。坏只在 **hybrid/recurrent + 投机** 的组合上。

**为什么当年量到 hits=0**：那是 **docker nightly 0.29.1rc1 + 640K/seqs2** 口径；两个版本之间
`_warn_if_unannotated_eagle_mamba` 已被移除/改名 ⇒ 结论不可外推。受影响的旧文本已更正（8115/8116 启动脚本头注释 + ledger L36）。

**能否解决**：① 上游正路——给 Qwen3.5 实现 `mamba_cache_mode='all'`（大工程）；
② 外科手术——让 mamba 组参与 EAGLE 末块语义（小改，但递归状态回滚语义有静默算错风险，必须先独立数值验证）；
③ 现役绕过——单流关掉前缀缓存（107.8 t/s），要跨请求复用时按 workload 权衡（复用粒度 544 token/块）。

### 7decies. 事故与规程：两会话共用 8117 导致互杀（2026-09-18）

并行会话同跑同一个 launcher ⇒ **同一 PID 文件 `logs/ornith397b-8117.pid` 与同分钟日志名互相覆盖**：
对方 20:46 启动后覆盖了 PID 文件（我的 arm2 服务被对方停服逻辑杀掉、且对方的 `>` 重定向截断了同一份日志），
我 20:54:59 按共享 PID 文件停服时**误杀对方服务（pid 637790）**。我 arm2 的探针因此打到了对方的服务上（queries +0）⇒ **该臂数据作废**。

**机制化修复**（不靠自觉）：`quark-int8/_guard.sh`
- 本会话实验用**专用端口 8127**（已登记 `config/ports.conf`）⇒ PID 文件/日志名天然隔离；
- 起服前**硬门**：存在任何其它 `api_server` 进程、或任一 die 显存 <62 GiB ⇒ 直接退出并报告，绝不"腾地方"；
- 停服**只杀本会话 PID 文件记录过的 pid**，且杀前核对 `/proc/<pid>/cmdline` 里的 `--port` 是本会话端口。

### 7undecies. mamba_cache_mode='all' 实测否决；agent 多轮的正解是 align（2026-09-18）

**目标**：让逐条多轮（Claude Code / DSH 式）的 TTFT 不随轮数退化。

**三态多轮 E2E（4 轮 × 1500 tok 回答，同 prompt；专用端口 8127 隔离）**

| 配置 | 重复 prompt 命中 | 多轮命中 | 多轮 TTFT s | count TPS |
|---|---|---|---|---|
| 关 | 0 | 0 | 0.61–0.82（增长） | 107.49 / 107.80 |
| **align（已设为 8117 默认）** | **+2176** | 1632/3264/5984 | **0.60/0.61/0.60/0.32（平）** | 107.00–107.52 |
| all | **+0** | 0/0/0/2048 | 0.57/0.75/1.14/1.10（增长） | 89.83 |

**为什么 'all' 不通**：它要求"每个 block 边界都有一份状态快照"，而 GDN 的融合 chunked 内核**每个 chunk 只产出末尾那一个状态**。
align 可用是因为它**强制 chunked prefill 且把调度步长对齐 block_size**（`config.py:622-624` 的 assert + 管理器对齐逻辑）⇒
每步末尾恰落在块边界，那份状态就是块边界快照。要真做 'all' 必须改内核（按块边界吐中间状态），
而它能补的只是 align 吃不到的"上轮**生成段**内快照"——实测每轮 0–1.6k token（20k 上下文下约 0.2–0.5 s/轮）。
另：`all` 在 262144 下单请求需 **16.47 GiB** 状态内存（可用 5.95）⇒ 引擎 `ValueError` 起不来；
`maxlen 65536`（4.12 GiB）或 `mamba_block_size 2176`（4.12 GiB，粒度变 2176）才装得下。

**数值安全（这是敢开前缀缓存的前提）**：`reuse_consistency.py` 硬门——同一 prompt 冷算 vs 命中复用，
**输出 token 序列逐位相同、`|Δlogprob| = 0.000e+00`**；align 与 all 两种模式均 PASS。

**落地**：8117 默认改为 `--enable-prefix-caching`（align）。依据是本臂 align 代价≈0（107.00/107.30/107.48/107.52 vs 关 107.49/107.80）
而多轮 TTFT 由"增长"变"平"；KV 池 369,503（关：371,720，−0.6%）。int4 臂的 +3.7%（关更优）结论不变，两臂不冲突。
补丁 `patches/gfx90a/port_mamba_all.py` 保留但**默认不启用**（需 env + `--mamba-cache-mode all`），一条命令可 `--revert`。
