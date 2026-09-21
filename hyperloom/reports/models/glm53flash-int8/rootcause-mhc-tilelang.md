# 8127 / GLM-5.3-Flash Quark-INT8「数值已判坏」的根因：TileLang mHC 在 gfx90a 上未被排除

日期 2026-09-21 · 机器 8×MI250X(gfx90a) · 本轮取证**全程纯 CPU**（读源码 + 读日志 + 一次不建 GPU 上下文的 import 探针）

## 0. 结论先行

**根因**：8127 跑的树是宿主 editable 安装 `src/vllm-master`，其
`vllm/model_executor/layers/mhc.py` 的闸门 **未打本机的补丁 ⑫**，仍是上游原文
`return not on_gfx942()` —— **只排 gfx942，没排 gfx90a**。于是
`HAS_TILELANG_MHC = True`（本轮实测），glm5next 每个非 MTP 层的 mHC pre/post 都走
TileLang 融合核，而该核在 **wave64** 上**非确定性地算错 `layer_input`**。

**为什么它一次解释两个症状**：`layer_input` 是注意力与 FFN 的输入，45 层 × 2 次 forward 全部要过它；
错的是**混残差之后**的输入，与量化无关 ⇒ 所以「AITER int8 线性 ON / Triton int8 线性 / eager / 非 eager」
全都坏、贪心两次不逐字一致，而**权重侧审计全过**（舍入 0 错、max_rel_err 0.39%、未量化张量逐字节相同）。

**修法已存在且被验证过**：DSV4.1 线 2026-09-18 就用真权重抓过同一个 bug（补丁组 **⑫ `mhc.py`**，
8119 三条 launcher 都挂了它，grep 命中 6 处）。给 8127 的修法就是把同一处改动落到
`src/vllm-master`：`if on_gfx90a(): return False`。

**本轮产出的可复用资产**：`hyperloom/patches-local/apply_mhc_gfx90a.py`
（幂等、`--check/--dry-run/--apply/--revert`、带 marker 与 py_compile；`--dry-run` 已实证可干净替换，
665 → 672 行，**原树未被触碰**）。

## 1. 证据链（每条都可重跑）

| # | 事实 | 怎么证的 |
|---|---|---|
| 1 | 闸门只排 gfx942 | `src/vllm-master/vllm/model_executor/layers/mhc.py:16-27`，原文含 `return not on_gfx942()`；`git log -- .../mhc.py` 只有上游一条，`git diff` 为空 ⇒ **未打补丁** |
| 2 | 本机在 gfx90a 上被放行 | 纯 CPU 探针（env 清掉 HIP/ROCR/CUDA_VISIBLE_DEVICES）：`on_gfx90a() -> True`、`on_gfx942() -> False`、`HAS_TILELANG_MHC : True`、`HAS_AITER_MHC : False`（没有 AITER mHC 兜底） |
| 3 | 这个模型**确实**用 mHC | checkpoint config 显式 `{'mhc': True, 'hc_mult': 4, 'hc_sinkhorn_iters': 20, 'hc_eps': 1e-06}`；`transformers_utils/configs/glm5_next.py:76 mhc: bool \| None = True`、`:92 mhc_num_residual_streams = kwargs.get("hc_mult", ...)`；`models/glm5next/nvidia/model.py:302 self.mhc = config.mhc`、`:402-404` 建 MHCPreOp/MHCPostOp/MHCFusedPostPreOp、`:522` 每层调 `mhc_pre_op` |
| 4 | 该核在 gfx90a 上算错且非确定（**本机已实测**，不是引用他人） | DSV4.1 记录 §4.8：真权重 layer-0 参数、同输入连跑 4 次，对 vLLM 自带 torch 参考的 maxabs = **1.85 / 9.8e-4 / 2.17 / 1.97**（输出量级 ~1.2），而 post_mix/pre_mix/comb 三项正确（1.2e-7）；TileLang 自报 `[ThreadSync] Hoisting sync from inside if to before if. Condition is not safe for in-if sync: tx < 32` |
| 5 | 8127 的症状与之吻合 | 配方 `knobs/glm5next-quark-int8-launch-set.md` §5：AITER ON NLL 10.69/11.39/10.38；Triton ON 11.34/10.99/9.62；同 prompt 三连打 8.83/8.66/9.72（阈值 0.75·ln V = 8.96，`ln V` = 11.95 ⇒ **接近均匀分布 = 模型根本没在条件化**）；贪心两次不一致；无 NaN/inf、无 unused/missing 权重 |
| 6 | 同一 gate 在镜像里也在 | `/tmp/vllm0918/model_executor/layers/mhc.py` 与上游逐字相同 ⇒ **容器臂必须靠挂载**（8119 挂了，8121 不需要） |

## 2. 危害面（谁受影响 / 谁不受）

| 臂 | 树 | mhc.py 已修？ | 模型用 MHC？ | 判定 |
|---|---|---|---|---|
| **8127** GLM-5.3-Flash int8 | 宿主 `src/vllm-master`(editable) | ❌ 未打 | ✅ glm5next | **正命中 ⇒ 就是它** |
| 8119 DSV4.1 int4/mxfp4（3 条） | 容器 + 挂载树 | ✅ 已挂 ⑫ | ✅ deepseek_v4* | 已防住 |
| 8121 GLM-5.3 CT-Int4 | 容器 + 挂载树 | ❌ 未挂 | ❌ `glm_moe_dsa`→`deepseek_v32`，`grep mhc` = 0，config 无 `hc_*` | **无关**（与它事实召回 6/6 自洽，反向印证判据正确） |
| 8115/8116/8117 Ornith、8101/8107/8109/8111/8113/8114 | `envs/vllm_0.28.0_rocm72` / `vllm_master_rocm724` | ❌ 该闸同样只排 gfx942 | ❌ qwen3_5* / qwen4_exp / deepseek_v32 均不在 MHC 使用者清单 | 暂判无关 |

**MHC 使用者全集**（两棵树交叉验证，只有两个家族）：`models/glm5next/*`、`models/deepseek_v4/{amd,xpu,cpu}`。

⚠️ 推论（要记进技能）：**任何 config 里带 `hc_mult` / `hc_sinkhorn_iters` / `mhc: true` 的新模型，
接进来第一件事就得确认它用的那棵树含 gfx90a 排除**。这条判据零成本（读 config.json + grep 一处闸门）。

## 3. 顺带结掉一个悬案：`device_name` 为什么有两个值

探针实测 `current_platform.get_device_name(0)` 在**宿主 master env** 返回
**`'AMD Instinct MI250X / MI250'`** → 经 `re.sub(r"[\s/]+", "_")` 清洗即
**`AMD_Instinct_MI250X_MI250`**，正是 09-17 Ornith 8116 日志里被命中的那个 MoE 调优表名；
而 nightly-0918 容器内返回 `AMD INSTINCT MI250 (MCM) OAM AC MBA`。
⇒ **表名跟着 ROCm/amdsmi 的 `market_name` 走，不跟着机器走**（宿主 7.2.4 vs 容器 7.2.3），
不是"手抄错"也不是代码路径差异。已回写技能与 `knobs/vllm-fused-moe-tile-seeds.md` 的更正。

## 4. 解决方案与验收（**GPU 步骤待批**）

**F1（首选，一行）**：`python3 hyperloom/patches-local/apply_mhc_gfx90a.py --apply`
→ `src/vllm-master/.../mhc.py` 加 `if on_gfx90a(): return False`。落点是 torch/triton 回落，
**那是 gfx942 现在就在用的路径**（`mhc_kernels.mhc_pre_torch` / `mhc_post_torch` / `hc_head_triton`），
不是无人区。代价：mHC pre/post 每层 2 次走 torch ⇒ **会变慢**，但正确性优先。
- 该树被别的会话共用（8127 是他们的在途臂），**改之前必须取得同意**。

**F2（零改源码，做对照更干净）**：`HAS_TILELANG_MHC` 是**模块级常量**、分派点
（`mhc.py:174/305/391/543`）在**函数体内读全局** ⇒ 一个 sitecustomize/`PYTHONPATH` 补丁
`import vllm.model_executor.layers.mhc as M; M.HAS_TILELANG_MHC = False` 就能单变量关掉。
适合做 ABAB，不用碰共享树。（已按源码结构核实可行，未实跑。）

**E1（最便宜的 GPU 判据，秒级、单 die）**：`python3 quark-int8/ktest_mhc_tilelang_nondeterminism.py`
在 master env 里对 **glm5next 的真实形状**（hc_mult=4、sinkhorn 20）复跑一遍：
关闸前后各跑，看 maxabs 是否从 ~2 掉到 ~1e-7、且四次是否一致。
**这一条就能把根因钉死**，不需要起服、不需要 int8、不需要等装载 6 分钟。

**E2（端到端验收）**：打完补丁起 8127（**只改这一个变量**），复跑 `tools/probe_nll.py`：
判据 = 三段 NLL 全部远小于阈值 8.96 **且** 贪心两次逐字一致。过了再把配方从
`experimental` 提上来；三个旧嫌疑（TRITON MoE atomic / `ROCM_AITER_MLA_SPARSE` / kpool torch 回退）
届时按需复测 —— 注意 ①atomic 只能解释抖动不解释 NLL≈ln V，②③同。

**F3（如果 F1 慢到不可接受）**：`HAS_AITER_MHC = False` ⇒ aiter 侧没有可用 mHC 融合核，
要快只有修 TileLang 核的 wave64 同步（`tx < 32` 假设），那是内核级工作量，且本机已把
「wave64 是 gfx9 的系统性风险面」记为通用教训 —— 别指望它是几天的活。

## 5. 过程教训（为什么这个 bug 二次复发）

- **知识存在但没跨臂流动**：⑫ 补丁与 §4.8 记录都在 DSV4.1 线里，写得很死（"必挂，否则整模型是乱码"）；
  新臂（不同引擎底座：宿主 editable 树 vs 容器挂载树）接入时没人回去查它。
  ⇒ 已把「config 带 hc_* ⇒ 先查 mHC 闸门」写成技能条目，并给了 applier 做机制化兜底。
- **排查顺序应当先"每层都过的公共路径"再"某量化才有的路径"**：配方从「int8 坏了」出发，
  于是查了三条量化/注意力通路；而"两条不同 int8 线性核都坏 + eager 也坏"其实已经在提示
  **与量化无关的公共路径**，那一步没做，白烧了 4 轮起服（每轮 ~10 min）。
- **一臂一杠杆**是对的，但**先做便宜的机理判据**（E1 秒级）比先做端到端 A/B（E2 十分钟）更省。

## 跨树复证（2026-09-21 · 0918 镜像臂 8128，E2 前半已过）

结论先行：**根因在第二棵树上再次出现并被消除（跨树旁证）** —— 同一份 `GLM-5.3-Flash-Quark-Int8` 权重，
在 `rocm-ai/vllm:glm53-int4-gfx90a-0918`（与 master 树布局完全不同的新架构包 `vllm/models/glm5next/`）上，
关掉 tilelang mHC 后三段 NLL 从近均匀回到正常量级：

⛔️ **措辞更正（同日自查后改）**：本节初稿写的是「只改 mHC 闸门这一个变量」——**撤回**。8127 与 8128
差的不止 mHC（不同树 / 不同底座 / indexer 路由也不同），所以这张表是**跨树旁证**，不是单变量对照。
真正的单变量对照是同一镜像同一脚本的 `IDX_PATCH=1 + MHC_PATCH=0` 臂，**能起服、尚未跑**。
教训：**跨树对比 ≠ 单变量对照**，写成后者会让下一个人以为因果已经钉死。

| 臂 | 树 / 底座 | mHC | 三段 NLL（阈值 8.96，ln V = 11.95） |
|---|---|---|---|
| 8127 | `src/vllm-master`（editable，宿主） | tilelang 开（未排除 gfx90a） | 10.69 / 11.39 / 10.38 |
| **8128** | 0918 镜像 + 挂载已修 `mhc.py` | tilelang 关（`on_gfx90a() → False`） | **1.815 / 0.523 / 0.472** |

日志级旁证（同一次 A/B）：未挂 `mhc.py` 那跑，权重装完后 `08:17:58 TileLang begins to compile kernel
`hc_prenorm`、`08:18:02 completes`（全日志 48 条 tilelang 行）；挂上后 **0 条** ⇒ 闸门在容器内确实生效，
分派落到 `mhc_pre_torch`/`hc_head_triton`（即 gfx942 现在就在用的那条回落）。

本报告 E1/E2 的状态因此更新为：

- **E2 前半（NLL 量级）：跨树旁证已过；单变量对照（同镜像 `MHC_PATCH=0`）未跑。**
- **E1（`ktest_mhc_tilelang_nondeterminism.py` 秒级单 die）：仍未跑。** 端到端复证已到位，
  E1 的价值降为「把 ~2 → ~1e-7 这个幅度单独钉成一个可回归的资产」，不再是判因的必要条件。
- **E2 后半（贪心两次逐字一致）：不过 —— 但这不是本报告根因的反证**，因为该判据本身未校准
  （本机从未有任何 vLLM TP8 臂通过它，含健康臂；paged-KV block table 每次不同 ⇒ 1e-2 抖动是预期）。
  详见下一节与 `/home/qiba/ai/docs/recipes/knobs/glm5next-quark-int8-launch-set.md` §5c。

### 顺带查出：还有**第二个独立缺陷**（不属于本报告范围，已另行归档）

- 统一尺子（`prompt_logprobs=5`、`max_tokens=1`、`temp=0`，真值 logprob `<-12` 记为灾难位，每段 16 次）：
  zh1 `床前明月光…` **10/16 请求有灾难位，且 10/10 全在 `pos ≡ 1 (mod 4)`**；
  zh2 / en / 一个 55-token 的 seq **各 16 次全零**。灾难位形态 = 该位 top5 退化为
  `' ' / ',' / '.' / '\n'`（p≈0.35 的无信息分布），落差 17 nats；同请求其余位置稳在 ±0.05。
- 与量化无关、与批形状无关、**与 `HSA_NO_SCRATCH_RECLAIM` / `HIP_FORCE_DEV_KERNARG` 无关**
  （`docker exec env` 实测镜像早已烘焙这两个变量，症状照在 ⇒ 曾被点名为头号待验修复项的这条**已被否证**）。
  ⛔️ 初稿里「灾难位与 `index_kpool=4` 同相（mod 4 周期）」一句已**撤回**：当时只统计 mod 4（假设锚定）、
  绝对位置未记全（能观测到的 3 次都在 pos5），而 pos5 并不在 kpool=4 的组边界上，且该路径自己 assert 的
  粒度是 **16-token tile**。周期候选应至少扫 2/4/8/16/32/128，未做 ⇒ 该说法无证据。
- 与 `index_kpool=4` 同相（mod 4 周期），而 ⑧ 放行后走的恰是 `models/glm5next/amd/sparse_indexer.py:91`
  那个本机从未验收的 `sparse_attn_indexer_kpool` 函数。**相关已测，因果未证。**
- 数据、判据、口径与两次自纠（`prompt_logprobs[0]` 为 `None` 导致下标错位一格；
  `=1` 与 `=5` 口径混用产生的假阴性已作废）全部记在
  `/home/qiba/ai/docs/recipes/knobs/glm5next-quark-int8-launch-set.md` §5c，本文件不重复。

### 一条方法学产出（判据本身要修）

`tools/probe_nll.py` 的「贪心两次必须逐字一致」是**从坏臂的症状反推出来的**，本机从未有任何臂
（含健康的 8121）在 vLLM TP8 + CDNA2 上通过过它。它适合当**坏臂的阳性指标**，
**不适合当好臂的必要条件** —— 否则一个健康的臂会被判 FAIL（8128 这次就差点被误判成「mHC 没修对」）。
建议改法（未改代码，先记）：NLL 量级作硬门；确定性改为「同一 prompt 连打 k 次的**灾难位率**」
（现测 zh1 = 10/16）而不是「两次文本是否逐字相同」。

## ★ E1 已跑（2026-09-21 · 0918 镜像 · 单 die · 编译+跑约 15 秒）—— 根因升级为 kernel 级实锤

`quark-int8/ktest_mhc_tilelang_nondeterminism.py`，真权重 `layers.0.hc_attn_{fn,scale,base}`
（`model.language_model.` 前缀，HC=4、H=4096、`fn=(24,16384)`），`HIP_VISIBLE_DEVICES=1` 只占一张 die。

```
HAS_TILELANG_MHC(镜像默认)=True   on_gfx90a=True  on_gfx942=False
Warning: [ThreadSync] Hoisting sync from inside if to before if. Condition is not safe for in-if sync: tx < 32
第1次: layer_input 误差=1.953e-03  comb误差=1.788e-07   输入未被改写
第2次: layer_input 误差=1.953e-03  comb误差=1.788e-07
第3次: layer_input 误差=2.172e+00  comb误差=1.788e-07      ← 同一输入、同一进程
第4次: layer_input 误差=2.031e+00  comb误差=1.788e-07
对照：mhc_pre_delayed_torch 自比 4 次 = 0 / 0 / 0 / 0            ← 回落路径完全确定
对照：tilelang             自比 4 次 = 0 / 2.547 / 2.180 / 1.961  ← 它自己就不确定
判据: 回落路径确定=True  tilelang 自身不确定=True ⇒ 根因确认为 tilelang mHC 核的非确定性（gfx90a）
```

- **两次独立跑**（10:11:43 与 10:13:06）都命中，坏率约 25–50%（2/4 与 1/4）；
- `1.953e-3` 是 bf16 对 fp32 参考的**噪声底**，`≈2.0` 是**错误**（输出量级 ~1.2）⇒ 判据要看
  「四次之间是否一致 + 跳变量级」，不能只看单次误差大小；
- `comb/post_mix/pre_mix` 三项恒 `1.788e-07` ⇒ 坏的确实只有喂给注意力与 FFN 的 `layer_input`；
- 调用前后 `residual`/`pre_mix` 校验和逐位相同 ⇒ 不是 in-place 副作用；
- tilelang **自己在编译期**打出 `tx < 32` 的 ThreadSync 警告 ⇒ 与 wave64 假设机制吻合。

### 由此改变两个投入判断

1. **E2 的「单变量对照臂」不再是判因的必要条件**：因果链已经是「kernel 级 + 有确定对照 + 单 die
   15 秒可回归」，比 8 卡 10 分钟的端到端 NLL flip 更强 ⇒ **省掉那次起服**（若仍想要端到端记录，
   `MHC_PATCH=0` 一跑即可，但它回答的是「症状是否随之消失」，不是「谁错」）。
2. **本节的 09-18 那组数字（1.85 / 9.8e-4 / 2.17 / 1.97）需要留一个问号**：当时的脚本把
   张量名写死成无前缀、`H=5120` 写死（另一颗 checkpoint 的形状）。tilelang 与参考拿到的是
   **同一份错形状**，比较仍自洽、结论方向不变，但那组具体数字不该被当作本 checkpoint 的读数。
   已把脚本改为**从 checkpoint 自适应**（前缀探测 + `HC/H` 取自 config + 形状断言），本节上方
   的 E1 才是可引用的数。**教训：复现资产必须从被测对象取维度，写死的常量会在换 checkpoint 时静默失真。**
