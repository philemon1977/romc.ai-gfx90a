# 咨询：DeepSeek-V4.1-Flash 在 gfx90a 上输出退化为 token 复读，是 int4 量化损伤还是运行时组合 bug？

（所有数字都是本机实测，尺子在文末列出。已推翻的结论我会明确标"撤回"，并附真实原因。）

## 1. 环境与部署

- 硬件：8 × AMD Instinct MI250（gfx90a / CDNA2，双 die），**每 GCD 64.0 GiB HBM**，251 GiB 内存，48 CPU 核
- 约束：gfx90a **无原生 FP8/FP4 MMA**，AITER 的 MoE 路径不可用
- 软件：`vllm/vllm-openai-rocm:nightly`，`0.29.1rc1.dev47+gdc36fcce9`；TP=8、`--dtype bfloat16`、
  `--max-model-len 262144`、`gpu_memory_utilization 0.97`、`--enforce-eager`、`--tokenizer-mode deepseek_v41`
- 模型 DeepSeek-V4.1-Flash：40 个主干层（+3 个 MTP/DSpark 层，vLLM 里 `mtp.` 被 mapper 丢弃不加载）
  `hc_mult=4`（mHC / hyper-connections，`hc_sinkhorn_iters=20`、`hc_post_alpha=2.0`、`hc_eps=1e-6`）、
  `sliding_window=128`、`compress_ratios=[0,0, 2×18, 1×20, 0×3]`（即 layer0-1 无压缩、2-19 比率 2、20-39 比率 1）、
  `kv_source_layers=[2,8,14,20]`、`index_source_layers=[2,8,14,20,24,28,32,36]`、`index_topk=512`、
  `n_routed_experts=384`、top-6、`scoring_func=sqrtsoftplus`、`routed_scaling_factor=1.5`、`swiglu_limit=10.0`、
  `head_dim=512`、`qk_rope_head_dim=64`、`o_groups=8`、`o_lora_rank=1024`、
  ratio=0 层用 `rope_theta=10000`（普通 rope），ratio>0 层用 `compress_rope_theta=160000` + YaRN(factor=16, original=65536)、
  `engram_layer_ids=[1,14]`

### 权重格式（**关键**）

| | 官方发布 checkpoint（477 GiB） | 我在用的 CT-Int4（397.30 GiB） |
|---|---|---|
| 路由专家（模型主体，58% 的数值量） | **MXFP4 e2m1，group 32，E8M0 scale** | int4 g32，uint4b8（`value=(nibble-8)*scale`），bf16 scale |
| 注意力投影 + 共享专家 + indexer | **FP8 E4M3，block 32×32，E8M0 scale** | int4 g32，bf16 scale |
| engram 哈希表（2×94 GiB） | FP8 block32 | int4（我改的，省 97 GiB 才装得下） |

## 2. 症状

用法已确认正确（官方 `inference/generate.py` 默认 `thinking_mode="chat"`；checkpoint **不含 `chat_template`**，
vLLM 的 `deepseek_v41` tokenizer 用 `encoding/encode_messages` 的移植实现 `apply_chat_template`）。
`POST /v1/chat/completions`，`chat_template_kwargs={"thinking":false}`，`temperature=0`：

- 输出：`'\n 在 的 的 的 的 的 的 的 的 的 的 的 的 …'`（复读吸引子，`finish_reason=length`）
- **完全确定**：同一请求两次逐字节相同；解码 ≡ 预取（decode 与 prefill 结果一致）
- 对**每个**试过的提示都如此（中文、英文、事实题、翻译、指令）；`thinking=true/false` 相同
- logits 很"自信"（不是平坦分布）
- 单流吞吐 3.6 tok/s；KV cache 595,565 token；262144 上下文并发 2.27x（即服务本身是健康的）

## 3. 已按"权威 = checkpoint 自带 `inference/model.py`"实测**正确**的部分

| 部件 | 尺子 | 结果 |
|---|---|---|
| embedding 行 | 逐字节比 `embed.weight[ids]` | 相同 ✓ |
| layer0 注意力**输出** | 纯 torch 复现 rope / fp8 行量化 / attn sink / `o_groups` 分组输出投影 | **cos 0.999279** ✓ |
| mHC coefficients（pre/post/comb，含 20 轮 Sinkhorn） | 照抄 `kernel.py::hc_split_sinkhorn` 逐字复算 | post **2.4e-7**、comb **4.0e-7** ✓（并用 `post_alpha=1.0` 反证，误差 0.5 ⇒ 证明运行时用的是 2.0） |
| mHC post/pre 的应用方式（旋转链 A/B） | 复算 `y[j]=post[j]·x+Σ_i comb[i,j]·res[i]` | 3.2e-3 / 3.4e-3 ✓ |
| **MoE 端到端输出**（加权→累加→shared expert） | 真请求 `ffn_in` + checkpoint 权威语义 | **reldiff 0.5211%、cos 0.999964**（layer0）；0.5985%/0.999966（layer2）✓ |
| 路由 top-6 与权重 | 参考 gate | top-6 完全一致，权重和精确 1.5 ✓ |
| YaRN rope 表（layers 2–39） | 复算 | 1.2e-7（bit 级）✓ |
| compressor 池化+norm、压缩 cache 多行写读、`combine_topk_swa_indices` 槽位语义、engram 哈希 | 各自独立对拍 | 全过 ✓ |
| 全 40 层健康画像 | 逐层 rms 与"相对上一层的改动量" | 每层 rel **0.21–1.06** ⇒ **没有死层**，模型不是恒等映射 ✓ |

## 4. 我量化的"功能损伤"（这是我判断量化为主线的原因）

做法：拿真请求 dump 的输入，**同一套 gate**（⇒ 路由完全相同），分别用①源权重反量化=理想 ②我们的 int4，比输出。

- **layer 0 MoE 输出偏差 9.295%**（cos 0.995707）。分解：routed 自身偏差 15.986%、shared 自身偏差 8.918%；
  而**源权重下** routed rms=0.01914、shared rms=0.09410（**共享专家贡献是路由专家的 4.9 倍**）
  ⇒ 正交校验 `sqrt((0.0191×0.1599)²+(0.0941×0.0892)²)/0.0962 = 9.28%` ≈ 实测 9.295% ✓（分解自洽）
- **layer 0 注意力输出偏差 10.687%**（cos 0.994290）；注意力占该层后注意力的 **57.3%**（layer2 为 52.6%）
- 折算到"每层对残差流的扰动"：注意力 ~6.1% + FFN ~3.2% ≈ **6.9%/层**；40 层粗估累积漂移 `sqrt(40)×6.9% ≈ 44%`
- 权重层面：int4 vs 源 **relRMSE ≈ 10.0%、cos ≈ 0.9948**；且我们 `scale = absmax/7` 非最优
  （逐组 1 维搜索最优 scale 可到 **6.8%**，即 1.47×；最优倍数稳定落在 0.90）

### 正在验证的改进臂（尚未起服）

"路由专家 int4+最优 scale"＋"共享专家与注意力全部 bf16（不量化）" ⇒
实测 MoE 输出偏差降到 **2.050%**、注意力输出偏差 **0.000%** ⇒ 每层损伤 ≈ **0.7%**（6.9% → 0.7%，约 10×）。
代价：+1.14 GiB/rank ⇒ KV 只剩 ~0.84 GiB ⇒ **KV token ≈253,900 < 262,144，256K 上下文就不成立了**
（所以我这轮 A/B 把两臂都降到 `--max-model-len 65536`；判定上中性，因为评测最长 prompt 仅 ~600 token）。

## 5. 已排除 / 已撤回（避免你重复建议）

- **mHC 的 TileLang 内核**（曾是真实根因之一：`mhc_pre_delayed_tilelang` 在 gfx90a 上对同一输入四次跑出 maxabs 1.85/9.8e-4/2.17/1.97，且 TileLang 警告 `[ThreadSync] Hoisting sync from inside if…`）→ 已排除该后端走 torch，输出从"标点乱码"变成"能写出英文但复读" ✓
- **engram**：`ENG_OFF=1` 仍退化
- **SwiGLU 限幅**：真实 pre-activation |gate| 最大 0.57–0.81，远未触到 ±10；开关系无变化
- **提示框架 / chat 模板**：多种用法（含裸文本、官方 framing、thinking 两态）完全相同的退化
- **我们自己写的 int4 MoE GEMV 内核 / 我们的补丁**：MoE 端到端与参考差 0.52%，说明运行时算得对
- **撤回：我一度报"转换保真度 cos 只有 0.96 ⇒ 转换坏了"** —— 那是我自己脚本对 1180 万项做 float32 点积时的**相消失真**；`F.cosine_similarity`=0.9948、float64=0.9950，与既有校验日志一致 ⇒ **转换没问题**
- **撤回：我以为"压缩 KV 用了 MXFP4(E8M0/组32)，比参考粗"** —— 读码定案：`_rope_quant_insert_kernel` 只有 `bf16` 与 `fp8_e4m3` 两种落盘分支、**没有 fp4 分支**，而我们 `kv_cache_dtype` 默认 `auto` ⇒ 压缩 KV **按精确 bf16 存储**（比参考的 fp4 fake-quant **更准**）；窗口 K 是"448 维 fp8(组 64, UE8M0) + 64 维 RoPE 尾保留 bf16"，参考是"全 512 维 fp8(组 32, UE8M0)" ⇒ 同阶，不是崩坏源。**⇒ KV 侧整条通路不可能更差。**

## 6. 我的核心困惑（请你重点回答）

> **每一个能被单独测量的部件，与一个**独立权威**（模型自带的 `inference/model.py`/`kernel.py`）都对到 4e-7 ~ 0.52%；
> 但整模型输出退化成 token 复读。唯一量化出来的系统性偏差是 int4 权重误差（约 10%/张量、约 6.9%/层）。**

具体问题：

1. **约 6.9%/层（源于约 10%/张量 RMSE）的权重误差，是否足以把一个 40 层、带 hyper-connections 与 MoE 的模型推进"单 token 复读吸引子"？** 实践中"4-bit 压垮模型"的经验阈值大概在哪？注意官方源里路由专家**本来就是 4-bit（MXFP4 e2m1）**，而我做的是 `fp4 → int4-uniform` 的重量化 + `fp8 → int4`。
2. 官方把**注意力/共享专家放在 FP8**、把**路由专家放在 FP4**。把它们统一降到 **INT4 g32** 是否就是那个高风险动作？在显存只有 ~62 GiB/GCD 可用、且整模型必须 8 卡均分的前提下，有没有更划算的分配方案（比如按"该模块对输出的贡献幅度"来分配精度，而不是按参数占比）？
3. **在无法运行更高精度版本（源 477 GiB 且 gfx90a 无 fp8/fp4 MMA；若全部 dequant 成 bf16 则显存/KV 都爆）的前提下，怎样用一个便宜、可判定的实验区分"量化损伤" vs "运行时组合 bug"？** 我想过"按层做剂量-反应曲线"（只把前 k 层/后 k 层的稠密权重升 bf16，看质量何时跳变），你怎么设计最省？
4. **关于 mHC 的尺度问题**：实测残差流 rms 随深度**单调增长 70×**（layer0 = 0.098 → layer39 = 6.85），并且 FFN 子层输出 rms 从 0.095 涨到 5.48、在 layer15/39 出现尖峰。我的解释是：`ffn_norm` 之后 rms 恒等于 `ffn_norm.weight` 的 rms，而该权重本身随深度增大（layer0 = 0.1267 vs 实测 `ffn_in` = 0.1263；layer2 = 0.1457 vs 0.1457；layer15 = 0.6130 正是尖峰处），所以"增长与尖峰是 checkpoint 自身参数决定的，不是 bug"。**这个推理站得住吗？** 对一个训练好的 mHC 模型，70× 的残差流增长正常吗？如果正常，为什么每个 sublayer 输入都过了 RMSNorm，输出幅度却会随深度放大？
5. **贪心 + 完全确定 + 对所有提示同一吸引子**，这个组合更像"数值噪声压垮模型"，还是更像"某个全局性代码错误"（例如跨层的 `pre_mix` 传递、最终 collapse、或 head/final-norm）？
6. vLLM `deepseek_v4_1` 的 AMD 路径上，以下有没有已知问题：`should_ignore_layer(..., use_fnmatch=False)`（我本地已改成 True，否则 `ignore` 里的通配全不生效）、混合"量化/未量化"下的融合模块（`gate_up_proj←[w1,w3]`、`fused_wqa_wkv←[wq_a,wkv]`）、`fp8_ds_mla` 与 bf16 压缩 KV 布局、`sparse_attn` / `combine_topk_swa_indices` 在 gfx90a 的实现？

## 7. 尺子与可复现物

`ref_moe_end2end.py`（MoE 端到端）、`ref_mhc_mixes.py`（sinkhorn coefficients）、`quant_damage.py` / `quant_damage_attn.py`（功能损伤，以源权重为理想）、
`ref_layer0_attn.py`（layer0 注意力参考，cos 0.999279）、`verify_ct_int4.py`（全量转换保真度：3105 样本，cos 0.994–0.997、relRMSE≈0.10）、
`validate_opt_scale.py` / `int4_scale_optimality.py` / `measure_opt_step.py`（scale 最优性与步长取舍）、
`verify_ignore_consistency.py`（`classify ⟺ ignore` 静态一致性证明）、`verify_shard_types.py`（逐分片结构 QA）、
`eval_quality_ab.py`（字符级退化判据：`max_repeat_run`、`distinct_ratio`，阈值 run≥8 或 distinct<0.15，已用中/英正常与退化样例标定）。
完整记录：`docs/DeepSeek-V4.1-Flash-CT-INT4-W4A16-转换记录-2026-09-18.md`（§4.13–4.23）。

## 8. 一句话版本

> 8×MI250 上跑 DeepSeek-V4.1-Flash 的 compressed-tensors INT4 W4A16 checkpoint，服务健康、贪心解码完全确定，
> 但**所有提示都输出 `的 的 的 …` 复读**；逐部件对拍模型自带参考实现全部到 4e-7~0.52% 一致；
> 唯一量出的系统性偏差是 int4 权重误差（10%/张量 → 折算约 6.9%/层）。
> **问题：这更像量化损伤还是运行时组合 bug？怎样用最便宜的实验把它判死？**
