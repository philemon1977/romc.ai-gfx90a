# GLM-5.3-Flash Quark-INT8 的 Hyperloom 8 小时会话（2026-09-21 起）

**状态**：会话运行中。**本文件此刻只登记"环境、口径、门禁、事故"，不登记任何性能结论**——
所有 tok/s 数字必须等 `state.json` / `reports/final.json` 落地后按同一把尺子读，
不许用中途快照抢答。

## 0. 目标与口径

- 模型：`/mnt/stripe-3mix-3t2/models/ZhipuAI/GLM-5.3-Flash-Quark-Int8`（`Glm5NextForConditionalGeneration`，308.65 GiB）
- 诉求来源：用户实测**单流 decode 3.18–3.33 tok/s「太慢」**，要求用 Hyperloom 调优，**8 并发**、**8 小时**。
- workload 身份（工具自己写出的 canonical id，可据此回查 KB）：
  `inference:glm-5.3-flash-quark-int8:mi250x:vllm:glm5_next:glm5nextforconditionalgeneration:0.3.1.dev85+gdee37d891:int8`
- 启动计划（`quark-int8/scripts_local/hl_workload_flash8h.env`，权威副本入库）：
  TP=8 EP=1 CONC=8 ISL=512 OSL=512 PRECISION=int8 MAX_HOURS=8 TARGET_GAIN=40%
  OPT_FLAGS=`--gpu-type mi250x --max-model-len 8192 --max-minutes-framework-pct 0.45
  --max-minutes-kernel-pct 0.35 --max-minutes-sweep-pct 0.01 --no-enable-conc-sweep
  --extra-env NUM_PROMPTS=16 --extra-env VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256`
  SERVER_ARGS=`--dtype bfloat16 --max-num-seqs 8 --block-size 128 --kv-cache-dtype bfloat16
  --language-model-only --safetensors-load-strategy eager --max-num-batched-tokens 2048 --enforce-eager`

## 1. 规模是**反推**出来的（这是本轮最容易翻车的地方）

runner `vllm_mi250x.sh` 的 `--num-prompts ${NUM_PROMPTS:-$((CONC*10))}` 与 Hyperloom
`_workload_envs.py:1466-1475` 的 factor 规则（`seq_cost<=1024 => 10`）合起来会给本 workload
**80 条 × 512 输出 = 4.1 万 token**。按本机当前聚合吞吐（估 ~13 tok/s，见下）单候选测量就要 ~52 分钟，
8 小时只够 baseline + 7 个候选——测量本身吃掉了搜索。

- 用 `--extra-env NUM_PROMPTS=16` 钉到「8 并发两个整波」（`is_allowed_variant_env_key('NUM_PROMPTS')=True`
  已实测；`--extra-env` 现在对 vLLM 路径也会落进 `benchmark.envs`，源码注释写明这是修 MTP pin 时改的）。
- decode 速率对上下文**不敏感**这条本机已实测（ISL 64→16384 恒 3.18–3.33 tok/s，
  因为 `index_topk=2048` 每层只读 2048 个 key），所以缩短 ISL 不损失代表性。
- 装载侧：308 GiB 每候选重装一次，实测 342–808 s（0.38–0.90 GiB/s，ZFS ARC 125.9 GiB < 权重 ⇒ 结构性冷读放大，
  抬 `zfs_arc_max` 需 root，本机无）。这一项**不在** Hyperloom 能调的面上，是预算的固定开销。

## 2. 环境改造（全部纯 CPU，先做完才碰卡）

`hyperloom-local` 原容器**不能**直接服务本模型，三处硬缺失：

| 缺失 | 证据 | 处置 |
|---|---|---|
| 模型不可见 | 原容器只挂 `/home/qiba/ROCm.AI`(rw) + `/mnt/kioxia-cm6-3t8/ai/models`(ro)，容器内 `ls /mnt/stripe-3mix-3t2/...` = No such file | commit 后带 `--device /dev/kfd --device /dev/dri --group-add video --shm-size 64g` + stripe 只读挂载重建同名容器；旧容器改名 `-prev` 不删 |
| 缺第 ⑧ 件（glm5next indexer 放行 gfx90a） | 容器内 `models/glm5next/amd/sparse_indexer.py` 与 `-0918` 底座 **31781 B 逐字节相同** ⇒ baseline 必在第一次 `_dummy_run` 撞 "…only supported on AITER." | `docker cp` 补丁树那份（768 行，marker=1） |
| 缺 mHC gfx90a 回退 | 同法对拍 `mhc.py` = 29412 B 与底座相同 ⇒ tilelang 融合核算错 `layer_input`（见 `rootcause-mhc-tilelang.md`） | `docker cp` 990 行那份 |

产物镜像 **`rocm-ai/vllm:glm53-int4-hl-fl1`**（`sha256:1c803cf7…`）。
**为什么走 commit 而不是 Dockerfile**：可写层里有 Magpie 的 mi250x runner 注册
（`apply_mi250x_runner.py` 在容器内跑过），从底座重新 build 会丢。脚本：
`quark-int8/scripts_local/hl_flash_container.sh`。

## 3. 门禁结果（逐条可复查）

| 门 | 结果 |
|---|---|
| IR-2 `install.sh` | **INSTALL_RC=0** |
| mi250x 三处 applier（install.sh 之后重打） | runner + identity 都报 already/PASS |
| 8 条静态自检 | **全 PASS**（gpu_identity 104 CU / 不折叠 runner / HW_SPECS 含 mi250x / Magpie 有 `vllm_mi250x.sh` 且已注册 / image_selector gfx90a→mi250x / workload 钉了 NUM_PROMPTS） |
| `--gpu-type mi250x` 合法性 | 用**非法值**触发 argparse，choices 里含 mi250x ⇒ PASS（见 §4 事故①） |
| LLM 网关门（本轮新增） | `GATEWAY_OK`（见 §4 事故③） |
| IR-1 preflight | `model_path_ok` + `torch_cuda_device_count=8` + `foreign_serving_processes=0` + 8 卡各 10 MiB ⇒ **PASS** |
| 起服后健康检查 | `optimizer_alive=true`，session `20260921T133345Z-48643eb0` |

## 4. 三次自伤（都留痕，因为下一个人会重复踩）

1. **「CLI 接受性探针」用合法值 = 直接开了一次真会话**。原以为
   `optimize --gpu-type mi250x --model /tmp/nope` 会"报到模型路径就退"，实际是 argparse 全过 →
   会话目录 `session/nope-such-model/` 建出来 → Claude 编排 agent 起来（还按默认 2h 预算走），
   跑了 3.5 分钟才掐。⇒ 报告里那条 V2 判据只在**没 source .env** 的裸环境安全。
   已改成"传不可能的值读 `invalid choice`，断言目标词在 choices 里"。
2. **`pkill/pgrep -f` 自杀 ×3**：模式串出现在自己的 `bash -lc` 命令行里 ⇒ 匹配到自己，
   命令 rc=137 且"看起来没杀掉东西"。正解是字符类写法 `bundled/[c]laude`，或先 `pgrep` 拿 PID 再逐个 `kill`。
3. **LLM 网关 scheme+路径双错 ⇒ 静默空转**（最贵的一次）。`.env` 写
   `ANTHROPIC_BASE_URL=https://192.168.100.127:8107/anthropic`，真身是宿主上的 LiteLLM
   （`ai/envs/litellm_1_88`，只听 `192.168.100.127:8107`，`/v1/messages` 在根下）：https 报
   `SSL: WRONG_VERSION_NUMBER`、http+`/anthropic` 报 404。症状**不是报错**而是 SEED turn 挂死、
   coordinator 活着、零候选 —— 8 小时会在"看起来在跑"里烧完。已改成
   `http://192.168.100.127:8107`（备份 `.env.bak-gatewayfix`），并在链脚本里加
   **launch 前的网关门**：用 SDK 真正调用的那个 bundled CLI 带 `--model` 实打一次 `GATEWAY_OK`。
   （裸 python 探针不够：不带 `--model` 时 CLI 会偷偷用默认模型名去撞一个不存在的模型。）

## 4b. 会话前的一轮单变量实验：装载策略（结论已改掉配方）

起因是用户指令「用 hip ais 的方式直接加载到 vram」。先给**不可行**的证据：
在 `hyperloom-local`/fl1 容器的 vLLM 树里，`VLLM_AIS_DISABLE`、`VLLM_FST_KEEP_NOGDS`、`hipFile`、
`cuFileBufRegister`、`hipAmdFileRead` **命中文件数全为 0**（`fastsafetensors` 为 5）。AIS 是
`patches/qwen4exp/local-nogds-gds-for-tp-gt1.patch` 打在**我们 master 树**上的东西，不在 0918
nightly 底座里；而它存在的那些臂，本仓判词是「serve + AIS ⇒ `cuFileBufRegister` 5014 以崩代降，
三块盘交叉证明与盘无关、只属 `api_server` 进程树」——Hyperloom 起服恰是那个进程树。⇒ 不走。

但目标（少过宿主、砍装载时间）在这棵树里有等价杠杆：**不给** `--safetensors-load-strategy eager`
（= vLLM 默认 lazy，`default_loader.py`）。同镜像(-fl1)、同 `--max-model-len 8192`、同 TP8、同 blk128：

| 尺子（都取 vLLM 自带日志同一行） | eager | 默认 lazy（**同容器**，会话 #3 baseline） |
|---|---|---|
| `Loading weights took`(`default_loader.py:430`) | 805.24 s | **746.35 s ⇒ 仅快 7.3%** |
| vLLM 宿主 RSS 峰（12 s 采样） | 90.4 GiB | **36.2 GiB ⇒ 降 2.5×** |
| `MemAvailable` 最低 | 14.1 GiB（离 Ray 误杀线 12.6 GiB 只差 1.5 GiB） | **75.9 GiB** |
| `GPU KV cache size` | 1,041,873 tokens / 127.18x | **逐字相同** ⇒ 未扰动显存布局 |

⚠️ **撤回**：本节初稿写「lazy `Loading weights took` 444.47 s ⇒ 快 1.81×」。那个 444.47 s 来自
**另一个容器**（臂容器 `glm53flash-0918`，`server-8128-20260921-225717.log`），拿它去比
`hyperloom-local` 里的 805.24 s 正是本页禁令说的「跨容器对比 ≠ 单变量对照」。补上同容器的 lazy
数据（746.35 s）后时间收益只剩 **7.3%**，**1.81× 作废**。

仍然成立、而且更有用的是两件事：
① **宿主 RSS 降 2.5×、`MemAvailable` 最低从 14.1 GiB 抬到 75.9 GiB**，且两个容器里都复现
   （臂容器 37.5 GiB / 本容器 36.2 GiB）⇒ 是路径级现象，不是偶然；
② 因此「0918 上 eager 不必要」的理由换成**把装载从 Ray memory monitor 的误杀线上摘下来**，
   而不是「快一倍」。

未解释的余量：同一条 lazy 路径，臂容器 444 s vs `hyperloom-local` 746 s（1.7×）。同镜像、同权重、
同 blk128，已知唯一差别是后者同时跑着 Ray + coordinator + Magpie。候选解释是并发内存占用引发
ZFS 回收抖动，**未证**。⇒ **本机装载时间的主导变量不是 loader flag**，别再把它当杠杆排序。
**推翻的记录**：`knobs/glm5next-quark-int8-launch-set.md` 里「eager = 必给」原判词的实验对象是
8114 的 **0.28 wheel** 底座（症状=去掉后在 shard 0/2 静默死亡）。同页姊妹判词「AIS/FST 两变量在
0.28 wheel 全树 grep=0」已经说明两棵树装载路径不同 ⇒ 该判词**不可跨底座引用**。已在原处留痕改写，
未静默覆盖。

**数值侧验收**：默认 lazy 起来后 `tools/probe_nll.py` 三条量级全过，第 1/3 条与 eager 那跑几乎同值
（1.811/0.481 vs 1.815/0.472）；第 2 条（中文诗句）在**同一台服务器**上连打三次是 1.514/0.607/1.514
⇒ 属 batch/paged-KV 组成的 run-to-run 抖动，不是装载差异。探针那句「判定 FAIL」来自「贪心两次必须
一致」这条**未校准判据**（本机从未有 TP8 臂通过，含健康的 8121），按本页纪律它是阳性指标而非必要条件，
不采信。

（旧「边界照实说」段已被上面的撤回块取代：它当时把时间收益写成「1.81×，只给 ±10% 置信度」，
实际同容器复测只剩 7.3%——问题不在置信度区间，在于对照对象选错了容器。）

**未测的第三个变量**：`--load-format fastsafetensors`（与本树的 FST 支持是两回事，8114 曾实测
25.92 s / 25.17 GiB/worker 量级）。它可能比 lazy 再快一截，但那是**下一轮的单变量臂**，不混进本次会话。
## 4c. 会话中段的实测进度快照（截至 4h45m/8h，工具自己跑出来的）

`baseline_tput = 23.4178 tok/s`，**优化栈仍为空（0 KEEP）**。候选清单（口径：同一 workload，
每候选一次全新起服，装载 ~12 min + 测量 ~6 min）：

| # | 候选 | 改动 | 结果 |
|---|---|---|---|
| v00 | cudagraph-full-decode-only | `--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":8}` | **段错误**：捕获成功（`100% 2/2`、`Graph capturing finished in 12 secs`），首个真实请求 `!!!!!!! Segfault encountered !!!!!!` → `Worker proc VllmWorker-1 died unexpectedly`；段错误前一刻是 `Triton JIT during inference: BuildPrefillChunkMetadataKernel.kernel` + `fused_moe_kernel`，与 breakable 档同一对内核签名 ⇒ **cudagraph 三档全灭**，`--enforce-eager` 是当前唯一解（已回写 knobs §2） |
| v01 | dsv41_kernel_plus_full_decode_graph | 同类 + 内核 env | **failed**（同族死法，`magpie_nonzero_invalid_measurement rc=1`） |
| v02 | aiter-int8-w8a8-linear | `VLLM_ROCM_USE_AITER=1` | 23.6（**+0.8%**） |
| v03 | prefill-batched-4096 | `--max-num-batched-tokens 4096` | 23.1（−1.4%） |
| v04 | async-scheduling | `--async-scheduling` | 23.1（−1.4%） |
| v05 | aiter-int8-full-atomic | AITER + `AITER_LINEAR=1`（MOE/MHA/MLA/TRITON_GEMM/CUSTOM_AR/FUSION 全 0） | **24.1（+2.9%）全场最佳，未过 KEEP 门** |
| v06 | aiter-int8-full-plus-async | v05 + `--async-scheduling` | 进行中 |

**必须记录的方法论事实**：工具在 7 个候选里 **3 个翻 `VLLM_ROCM_USE_AITER=1`**。本仓对 AITER 的
判词主体是 **MoE**（gfx90a 终案不可用）与 **RMSNorm 门不查 arch**；而 v05/v06 只开 **linear/a8w8**
这一条窄路 —— 它恰好也是 8127 那次「`Selected AiterInt8` + `import module_gemm_a8w8` + 首个请求
segfault」的同一条路，**但这次没崩还测出全场最高**。而本会话是 `--no-eval`：**没有任何精度门**，
所以一旦某个 AITER 变体越过 KEEP 门，它会以「validated gain」的身份进最终报告。

**已定处置（用户拍板）**：不打断会话；会话结束后立刻用
`quark-int8/scripts_local/verify_aiter_linear_arm.sh`（已写好，未跑）复现 v05 的有效环境并上两把
硬尺：`tools/probe_nll.py` 的 NLL 量级 + `scripts_local/taskcheck.py` 的事实召回。
数值坏 ⇒ 进死路清单，并在此声明「该 +2.9% 不可采信」；数值好 ⇒ 修正库内判词的适用面
（AITER-linear 与 AITER-MoE 不是一回事，必须分开写）。
## 5. 结果（收场后填，全部取工具自己写的权威文件）

会话 `20260921T151223Z-34588c9d`，06:59:36 自然收场（deadline 前 13 分钟），`stop_reason=sweep_done`。

### 5.1 权威数字

| 项 | 值 | 出处 |
|---|---|---|
| `baseline_tput` | 23.4178 | `final.json/baseline_tput` |
| `current_best.tput` | 24.1077 | `final.json/current_best/measurement/tput` |
| `cumulative_gain_validated` | **+2.9457%** | 同上 |
| `optimization_stack` | **1 条**：`env VLLM_ROCM_USE_AITER=1 + AITER_LINEAR=1`（其余 AITER 子开关全 0），`accuracy=None`、`gain=None` | `state.json` |
| 阶段耗时 | PRELUDE 7223.6s ／ FRAMEWORK **15953.4s** ／ KERNEL **4734.9s** ／ SWEEP 118.2s（合计 7h47m） | `phase_elapsed_totals` |
| 候选账目 | 11 个候选 → **1 KEEP / 10 REVERT**（journal 22 条） | `reports/optimization_journal.json` |

### 5.2 口径纠正（重要，会误导下一个读数的人）

`final.md` 把吞吐标成 **`23.4 tok/s/GPU`** —— **错**，它是**整台 8 卡服务器的聚合 output throughput**。三条内证：
① `final.json` 里 `total_throughput = 48.2154 = 2 × tput`，而 ISL=OSL=512 ⇒ `total` 是 in+out、`tput` 是 **output**；
② `tpot_p90_ms = 333.45` ⇒ 每流 3.0 tok/s × CONC 8 = **24.0**；③ 服务器日志 `Avg generation throughput: 24.0`。
⇒ 对外引用这份结果时必须写 **24.1 tok/s（8 路聚合，output only，@CONC=8）**，
否则与 MI300X 的 per-GPU 口径对比会**虚高 8×**。

### 5.3 本轮真正的发现：这是**每步固定开销**瓶颈，不是带宽瓶颈，而并发是没用掉的杠杆

单流实测 3.18-3.33 tok/s；本次 CONC=8 时 `tpot_p90=333 ms` ⇒ 每流 3.0 tok/s，
**每流只退化 7%，聚合却近乎线性到 7.5×**。⇒ 步时里"随 batch 增长"的部分很小，
大头是每步固定开销（对照 int4 那棵树 profile：稀疏注意力 36%／每层集合通信 19%／非专家 GEMM 19%）。
按 mi250x roofline，本 workload 距访存下限还有 **一个数量级**。

**由此产生两条明确的下一步**（都不在本轮口径内，故未做）：
1. **并发扫描** —— 本轮按用户口径钉死 `CONC=8` 且 `--no-enable-conc-sweep`。近线性扩展说明
   CONC 16/32 很可能直接翻倍聚合吞吐。**这是对"太慢"最可能有效的一档，优先级高于任何内核重写。**
2. `--load-format fastsafetensors` —— 装载路径的第三个变量（本轮只证伪了 eager 的必要性：
   同容器 805.24s → 746.35s，仅 7.3%）。

### 5.4 内核车道：81 分钟 0 次尝试（不是"试了没保住"）

| 证据 | 值 |
|---|---|
| `kernel_optimization_summary.json` | `by_kernel: []`，`failure_reason_breakdown` **九项计数全为 0** |
| `final.md` Highlights | `response from kernel_agent: kind=geak_e2e_done status=timeout` |
| 阶段切换原因 | `KERNEL_AGENT → SWEEP (reason=kernel_no_more_leverage)` |

⇒ GEAK 跑到 `runner_timeout=4877s` 超时退出，**一次内核 campaign 都没发起**。最可能的门槛
（未证，列为下一步要验的假设）：baseline 时 `TraceLens runtime patch unavailable for framework=vllm
; profile will omit annotation-only flags` + journal 里 `roofline` 记为 **REVERT** ⇒ 没有带形状的
trace，`trace_analyze` 提名不出候选内核，内核车道就没有输入。**下次跑内核车道前应先钉这条**。

另有一条运行期告警必须留痕：`alert sev=high agent orchestration silent for 7273s (threshold=300s)`
—— 编排 agent 静默 2 小时（与 PRELUDE 的 7223.6s 几乎等长），robustness 报了但未被处置。

### 5.5 唯一 KEEP 的数值验收（进行中）

+2.9457% 来自 AITER-linear，而 `--no-eval` ⇒ 它**从未被精度验证**（`accuracy=None` 是写在栈里的字面事实）。
验收方式与判据见 §4c；执行体 `quark-int8/scripts_local/verify_aiter_linear_arm.sh`（复刻 runner 的
GEMV／自研 indexer preamble，改 `--network host` 以便宿主尺子直打）。结论待填：

- NLL 量级门：?
- 事实召回门：?
- 判定：?
## 6. 读法与复现

- 状态读取：`bash quark-int8/scripts_local/hl_watch.sh`
- 权威读数：`bash quark-int8/scripts_local/hl_final_report.sh`


- `baseline_tput` = ?
- 通过验证的 `optimization_stack` = ?
- `cumulative_gain_validated` = ?（`state.json` 口径）
- 内核车道是否产出 KEEP / REVERT = ?
- 读法：`bash quark-int8/scripts_local/hl_watch.sh`；产物 `session/GLM-5.3-Flash-Quark-Int8/<sid>/reports/final.{md,json}`
