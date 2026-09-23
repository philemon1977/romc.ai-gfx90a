# Ornith-1.5-397B Quark INT8-Attn + AITER int8 线性 + MTP(5) · :8117（单流最快臂）

日期：2026-09-18（配方定稿）／2026-09-22（本报告整理入库）｜ 机器：8×AMD Instinct MI250X (gfx90a / CDNA2，64 GB/GCD)
运行时：native `envs/vllm_0.28.0_rocm72`（vLLM 0.28.0，TP8）｜ 口径：固定 prompt + **丢弃首个请求** + 中位数（n≥5），256 token greedy，`temperature=0 ignore_eos=True`

**一句话**：与 int4 出厂臂（8116）**同一份 INT8-Attn checkpoint**，只叠四个实测有效的杠杆（AITER int8 线性补丁 + `SPEC=5` + 前缀缓存 `align` + 关 torch.compile 落盘缓存），把单流从"只有 int4 的 42%"做成"**稳定超过 int4**"——本臂定位是**单流最快**，不是高并发/超长上下文。

> 深水量化 / 内核级分析（int8 GEMM 57% 地板、CK vs Triton、MoE 占比 46%、GEMV 内核 −26% 否决）在已入库的
> [`quark-int8/RESULT.md`](../../../quark-int8/RESULT.md)；本文件是**该臂的配方与工程实践索引**，
> 权威启动脚本是 `models/ornith-ai/launcher/ornith_1.5_397b_int8w8a8attn_aiter_vllm_rocm72_mtp5_256k_8117_ornith_mi250dx8.sh`
> （在 `/home/qiba/ai`，不在本仓库；其头注释是本臂最新、最全的活文档）。

## 0. 结论（先读）

| 配方（同一 checkpoint / 引擎 / 机器） | count t/s | explain t/s | step ms | 接受率 | KV 池 @262144 |
|---|---|---|---|---|---|
| 8115 原厂：TRITON int8 Linear + SPEC=1 + 前缀缓存开 | — | 37.73 | — | — | 418,083 |
| + AITER int8 Linear（SPEC=1） | — | 43.35 (+14.9%) | — | — | 418,083 |
| + SPEC=5 & 关前缀缓存（TRITON） | 86.09 | 63.27 | — | — | 387,448 |
| + AITER（SPEC=5），**载入编译产物**类 | 103.22 / 103.60 | 71.09 | 39.1–39.7 | 35.4% | 363k / 372k |
| **+ 关编译缓存（现场编译）＝本臂出厂态** | **107.84** | **78.29** | **38.7 / 38.3** | **64.6% / 40.0%** | **371,720** |
| 参照：int4 出厂（CT-W4A16 + SPEC=5 + no-pfx） | 89.29 | 63.12 | 48.4 / 48.1 | 66.4% / 40.7% | 1,538,260 |
| **本臂 vs int4（同会话背靠背）** | **+20.8%** | **+24.0%** | **−20.1%** | 基本相同 | 约 1/4 |

- **最保守口径**（两边都取"载入编译产物"类）：**103.4 vs 89.7 = +15.2%**，步时优势依旧 −18%。
- 质量：NLL 1.9710 / ppl 7.18（SPEC=3，两次复测逐位一致）vs int4 出厂 SPEC=3 的 1.9654 ⇒ **差 0.28%，速度收益基本白拿**。
- 逐级分解（explain，同一把尺）：8115 原厂 TRITON/SPEC1/前缀开 37.73 → +aiter(SPEC1) 43.35 (+14.9%) → +SPEC5&关前缀(TRITON) 63.27（= int4 打平）→ +aiter(SPEC5) 84.61 → 交付态 78.29–84.61（现场编译类；载入编译产物类只有 71.09）。
- **aiter 收益随 MTP 深度放大**：SPEC1 +14.9% → SPEC5 count +27% / explain +33.7%。深度越大，"1 verify 步 + SPEC 个 M=1 草稿步"里 M=1 步占比越高，而 CK int8 GEMM 在 M=1（decode/GEMV 形状）上对 Triton 优势最大。
- 代价：KV 池只有 int4 臂的 ~1/4（371,720 vs 1,538,260 @256k）⇒ **高并发 / 超长上下文仍归 int4 臂**。

## 1. 四个杠杆（缺一即没吃上）

1. **AITER int8 Linear 补丁**：只解除 CDNA2 的 arch 门，其余 sub-switch（MoE/RMSNorm/MLA/MHA/FP8BMM/…）全关，保持"只开 int8 Linear"这一件事。补丁自门控：仅当 `VLLM_ROCM_USE_AITER=1` 且 `…_LINEAR=1` 才装 hook ⇒ `VLLM_ROCM_USE_AITER=0 bash 本脚本` 即干净对照。
2. **`SPEC=5`**：MTP 深度是单流第一杠杆（int4 臂 SPEC 1→5 +69%、count 3→5 +16%）。机制同构：MTP 头在两臂都保持 bf16、都走 `use_eagle()` 那条草稿路。
3. **前缀缓存 `align`（`--enable-prefix-caching`）**：本臂同为 45×GDN 混合架构，开前缀缓存 ⇒ `mamba_cache_mode='align'`，多轮/agent 负载的命门（见 §4）。本臂实测单流代价≈0（align 107.00/107.30/107.48 vs 关 107.49/107.80，n=3 spread≤0.1%）。
4. **关 torch.compile 落盘缓存（`VLLM_DISABLE_COMPILE_CACHE=1`）**：见 §3。

## 2. 测量纪律（本会话最大教训，别省）

- **TPS 不是速度指标，步时才**：同一条臂里 count 与 explain 的 TPS 差 38%（108.3 vs 78.6），而 step ms 几乎相同（38.7 vs 38.3）——差的全是 MTP 接受率。只报 t/s 的 A/B 可能只是在比草稿命中率。
- **必须带 workload 标签**：count（封闭/可预测）与 explain（开放/低接受率）是**两个 workload**，MTP 接受率不同 ⇒ TPS 不同，跨标签比较无意义。报数必须写清 count 还是 explain。
- 口径：固定 prompt + **丢弃首个请求** + 中位数（n≥5）⇒ 重复间 spread 0.1–0.2%；撒盐 prompt 只能到 ±10%（内容变化会改 MTP 接受率）。
- 工具：`python3 ~/ROCm.AI/quark-int8/step_probe.py <port> <tag> 5 count|explain 256` 给 step ms 与 accept%；`measure_median.py` / `nll_probe.py`（自带 `nll>8.0` 拒绝门）。

## 3. torch.compile 落盘缓存：跨启动散布的真正来源（杠杆④）

- 同一 8117 配方，**载入磁盘产物** vs **现场编译**，count/explain 差 5%（count）/ 18%（explain）；而差异主项**不是内核速度**（step 39.1 → 38.7 ms，仅 −2.5%），而是 **MTP 接受率**（explain 35.4% → 44.6%）——两条编译路径数值不同 ⇒ 草稿命中率不同（载入 AOT 时 vLLM 会 `disable_guard_check()`，inductor 缓存目录也不同）。
- 实测（count / explain，n=5）：载入产物 103.22/103.60 与 71.09；现场编译 109.36/108.43 与 83.70。冷启代价：现场编译多 ~100–150 s（662 s vs 511–571 s）。故本臂默认取现场编译。
- ⚠️ **不能当普适规律**：int4 臂上该效应不出现（12 次启动两类都给 89.5–90.2）。机制未验证，本条只声称"本臂实测如此"，开关 `VLLM_DISABLE_COMPILE_CACHE=1`。
- ⚠️ 因此**跨类比较会虚高**：int8 的 109.36 属现场编译类、int4 的 89.8 属载入类。同类比见 §0。

## 4. 前缀缓存 align：agent 多轮的正解（2026-09-18 追加实测）

- 4 轮 × 1500 tok 回答（vibe-coding 形状）：align **开** ⇒ 命中 1632/3264/5984 token，TTFT **0.60 / 0.61 / 0.60 / 0.32 s（不随轮数增长）**；同臂 **关** ⇒ 命中 0，TTFT 0.61–0.82 s 且随上下文增长。
- 机制：align = 强制 chunked prefill 且调度步长对齐 block_size ⇒ 每步末尾状态正好是块边界快照，可被下轮复用；命中粒度 = 544 token/块。
- 数值安全：命中复用 vs 全量重算 **逐位一致**（token 序列相同、`|Δlogprob| = 0.000e+00`）。
- ❌ 另一条路 `mamba_cache_mode='all'`（每块都留快照）**已实测否决**：本 build 下命中≈0（多轮 0/0/0/2048），TTFT 反而增长，count 掉到 89.83 t/s；再加 262144 下它需要 16.47 GiB 状态内存（可用仅 5.95）⇒ 引擎直接起不来。详见 `/home/qiba/ai/docs/Ornith-397B-Mamba-All-模式实现方案-2026-09-18.md`。

## 5. 引擎级结论：直接迁移（与权重格式无关，无需重测）

以下都是 arch/引擎层开关，**与权重格式无关**，从别的臂直接迁移到本臂（本会话已逐条 A/B）：

| 开关 | 实测 | 处置 |
|---|---|---|
| `--disable-custom-all-reduce` | gfx90a 上 CUSTOM **静默算错** | **必须留** |
| cudagraph `PIECEWISE` | −8.7% | 别开 |
| `--max-num-seqs 1` | −4.4% | 别设 |
| `NCCL_MIN_NCHANNELS`/`MAX_NCHANNELS=1` | −12.1% | 别设 |
| `--async-scheduling` | −0.5% | 噪声内 |
| `--block-size 128/256` | −7.6% / −1.3% | auto 544 最优，**别手设** |
| `use_local_argmax_reduction` | +0.3% | 噪声内 |
| `fuse_allreduce_rms` | 本 build **无此 pass** | — |
| QuickReduce `QR=0` | 只 TP4 有收益 | 本臂 TP8 不设 |

## 6. 起服 fail-closed 门（只读判断，全部 fail-closed）

1. **端口**：`nc -z 127.0.0.1 $PORT` 被占 ⇒ 退出。
2. **PID 文件**：旧 pid 还活着 ⇒ 先停它再走（本机两次误杀事故 ⇒ **只认 PID 文件，禁 `pkill -f`**）。
3. **权重可读性**：int4 权重曾因 root:root 0600 ⇒ PermissionError 13（123 文件）；本臂先 `find … -name '*.safetensors'` 只读试读，有不可读 ⇒ 拒绝起服（`sudo chown -R $(id -un):$(id -gn) $MODEL_PATH`）。
4. **显存**：sysfs `mem_info_vram_used`（bytes）逐 die 算空闲，`< VRAM_FREE_MIN_GIB`（默认 62）⇒ 有主或不够，拒绝起服（强行起会 OOM）。
5. **解释器 / ROCm 库树 / 模型 / aiter 补丁** 四路存在性检查，缺一路直接退出。

## 7. 验收判据（缺一即没吃上杠杆）

```
grep -a 'Selected AiterInt8ScaledMMLinearKernel' $LOG   # 反向门：TritonInt8ScaledMM = 补丁未生效
grep -a '\[aiter\] import \[module_gemm_a8w8\]'    $LOG   # JIT 模块已命中缓存（首建 2310 s，已缓存 187.5 MB）
grep -a 'Using TRITON Int8 MoE backend'           $LOG   # int8 MoE 只有 TRITON（不能是 HUMMING/CPU/EMULATION）
grep -a 'GPU KV cache size'                       $LOG   # 期望 ≈ 0.37–0.42M tokens @262144
```
⚠️ 反向门：日志出现 `Selected TritonInt8ScaledMMLinearKernel` = 补丁/环境没生效（静默退回），explain 会掉回 ~63（= int4）。别把这个当"int8 不行"。

## 8. 质量门

- `prompt_logprobs` / `echo+logprobs` 在 **SPEC=5** 下返回近似均匀分布（NLL≈ln V≈12.9）⇒ SPEC=5 配方的质量**测不出来**。
- 质量必须在 **SPEC≤3** 量（本臂 SPEC=3 代理质量 NLL 1.9710），探针自带 `nll>8.0` 拒绝（exit 2）门。

## 9. 已否证 / 别搬（写进头注释，免得后人当 bug 查）

- **`VLLM_ROCM_SPLITKV_PA=1` 在本配方自动失效**：门的第 1 条是 `max_query_len == 1`，本臂 SPEC≥1 ⇒ 一步都不接管（等于没开）。想让它在**长上下文单流**生效必须另跑 `SPEC=0` 对照臂。
- **`MI250_GATE_GEMV=1`**：仍默认开（同构模型每层都有 shared_expert_gate），但该 +19.8% 是在 **35B** 上量的，**本臂从未单独 A/B** ⇒ 保留"未验证"标记，别当已验证结论。
- **自研 GEMV 式 MoE 内核（MI250_MOE_GEMV）**：真实调用形状（M=24/240 pairs）下**更慢**（261.4 vs 194.1 µs = −26%；只在 ≲80 pairs 才赢）。SPEC 越大 verify 步的 M 越大 ⇒ 对本臂只会更不利。
- **MI250X MoE 调优表（`VLLM_TUNED_CONFIG_FOLDER`）**：int4_w4a16 键下默认 16/64/32 = 89.81 已最优，8 个扰动全负；且本臂 MoE 走 TRITON int8，键都不同 ⇒ 单流没有值得写的表。
- **CK/aiter 的 int4/fp4 内核线**：本臂是 int8 oracle，内核空间完全不同。唯一可迁的是**方法**：内核收益要在独立 microbench 上量，不信端到端小差值。

## 10. a8w8 调优表补 gfx90a 行（消日志，性能恒等）

- aiter 自带的 `a8w8_tuned_gemm.csv` 只有 gfx942（26 行）/ gfx950（553 行），**零 gfx90a 行** ⇒ 本臂每次启动打 432 行 "not found tuned config … will use default config!"。
- 已用本机实测生成超集表（原表 579 行原样 + gfx90a 54 行），单路径注入 ⇒ 启动脚本默认 `AITER_CONFIG_GEMM_A8W8=$A8W8_GFX90A_CSV`。
- ⚠️ **补表不可能提速**（查证 + 实测）：这条路径是 `gemm_a8w8_CK`，从表里**只取 `splitK`**，而 gfx90a 上 `splitK=1..4` 全部 `RuntimeError` ⇒ 该列只有 0 一个合法值 = 现有默认值 ⇒ 行内容"正确"但**性能恒等**。价值只有两个：① 消掉每次启动 432 行 INFO；② 把本机实测 µs/bw/err 固化在表里。
- 该路径整块真实成本（图内 replay 口径，排除 Python/host）：90 次/步 = **0.99–1.06 ms/步 = 步时的 2.6–2.7%**；即使换成完美内核（1.6 TB/s 访存下限 0.41 ms/步）也只能拿回 ~1.5%。⚠️ eager 一次调用 ~46 µs（其中 ~40 µs host）⇒ 90 次/步 = 4.17 ms/步（步时 10.8%）——生产靠 cudagraph 躲掉，**再次支撑"绝不加 --enforce-eager"**。

## 11. 停服纪律（本机两次误杀事故的教训）

- **只认 PID 文件、绝不 `pkill -f`**：`kill -TERM -$(cat logs/ornith397b-8117.pid)`（负号 = 整个会话组）。
- ⚠️ **2026-09-18 事故**：两会话同跑 8117，PID 文件与同分钟日志名互相覆盖 ⇒ A 会话按 PID 文件停服**杀掉了 B 会话的服务**。规程：实验用 `PORT=8127` + 独立 PID + `quark-int8/_guard.sh`（起服前硬门：端口/PID/显存任一不满足直接退出，不腾地方）；**做实验请用 8127，不要用 8117**。
- ⚠️ **孤儿引擎占卡坑**（2026-09-22 现场）：`setsid` 保护服务不被工具中断打死，但 APIServer 猝死后 `EngineCore`+`Worker` 会被 reparent 成孤儿（ppid=1）仍占满 8 卡 HBM（~61.6 GiB/卡），而 PID 文件里的 pid 已死 ⇒ 本仓 `wait_gpu.sh`/fail-closed 门只看 PID 会误判"卡空/服务活"。处置：对 PID 文件里的进程组号（= 原 APIServer pid）发 `kill -TERM -<pgid>` 收整棵子树。

## 12. 路径对照（本机 ↔ 仓库内对应物）

| 本机路径（`/home/qiba/ai`， operational，不在本仓库） | 仓库内对应物 |
|---|---|
| `models/ornith-ai/launcher/ornith_1.5_397b_int8w8a8attn_aiter_vllm_rocm72_mtp5_256k_8117_ornith_mi250dx8.sh` | 本文件 + `quark-int8/RESULT.md`（深水量化/内核）+ `quark-int8/step_probe.py`、`measure_median.py`、`nll_probe.py` |
| `docs/Ornith-397B-INT8Attn-提速-经验迁移-2026-09-18.md` | 本臂经验迁移原文（§0–§3 的原始出处） |
| `docs/recipes/serving/8117-ornith-1-5-397b-int8-attn-aiter-vllm-tp8-mtp5.md` | 8117 配方卡（operational） |
| `recipes/patches/vllm/vllm_0.28.0_rocm72/aiter-int8/`、`aiter_a8w8_tuned_gemm_gfx90a.csv` | 已入库（`git ls-files`） |
| `quark-int8/int8aiter_arm{D,E,F,G}.sh`、`int4_armH.sh` | 三臂/交付态/回测 runner（在 `quark-int8/`） |
