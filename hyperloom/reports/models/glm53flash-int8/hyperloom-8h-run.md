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

| 尺子（都取 vLLM 自带日志同一行） | eager | 默认 lazy |
|---|---|---|
| `Loading weights took`(`default_loader.py:430`) | 805.24 s | **444.47 s（1.81×）** |
| `Model loading took`(`model_runner.py:419`) | 806.8–807.5 s | 452.2–452.6 s |
| engine init → `GPU KV cache size` | 14m27s | **8m32s** |
| vLLM 宿主 RSS 峰（采样 12 s） | 90.4 GiB | **37.5 GiB** |
| `MemAvailable` 最低 | 14.1 GiB（离 Ray 误杀线 12.6 GiB 只差 1.5） | **71.3 GiB** |
| `GPU KV cache size` | 1,041,873 tokens / 127.18x | **逐字相同** ⇒ 显存布局未被扰动 |

**推翻的记录**：`knobs/glm5next-quark-int8-launch-set.md` 里「eager = 必给」原判词的实验对象是
8114 的 **0.28 wheel** 底座（症状=去掉后在 shard 0/2 静默死亡）。同页姊妹判词「AIS/FST 两变量在
0.28 wheel 全树 grep=0」已经说明两棵树装载路径不同 ⇒ 该判词**不可跨底座引用**。已在原处留痕改写，
未静默覆盖。

**数值侧验收**：默认 lazy 起来后 `tools/probe_nll.py` 三条量级全过，第 1/3 条与 eager 那跑几乎同值
（1.811/0.481 vs 1.815/0.472）；第 2 条（中文诗句）在**同一台服务器**上连打三次是 1.514/0.607/1.514
⇒ 属 batch/paged-KV 组成的 run-to-run 抖动，不是装载差异。探针那句「判定 FAIL」来自「贪心两次必须
一致」这条**未校准判据**（本机从未有 TP8 臂通过，含健康的 8121），按本页纪律它是阳性指标而非必要条件，
不采信。

**边界照实说**：eager 侧跑在 `hyperloom-local`（Magpie 起服），lazy 侧跑在臂容器 ⇒ 跨容器，
时间差不是严格单变量；但秒数取自同一行日志、RSS/余量是路径级差异、KV 池逐字相同 ⇒ 「eager 在 0918
不必要」成立，而「快 1.81×」只给 ±10% 置信度——本机装载时间本身受 ZFS 缓存态支配（历史 eager 亦见
334 s 与 620 s 两次）。

**未测的第三个变量**：`--load-format fastsafetensors`（与本树的 FST 支持是两回事，8114 曾实测
25.92 s / 25.17 GiB/worker 量级）。它可能比 lazy 再快一截，但那是**下一轮的单变量臂**，不混进本次会话。
## 5. 结果（待填，禁止提前抢答）

- `baseline_tput` = ?
- 通过验证的 `optimization_stack` = ?
- `cumulative_gain_validated` = ?（`state.json` 口径）
- 内核车道是否产出 KEEP / REVERT = ?
- 读法：`bash quark-int8/scripts_local/hl_watch.sh`；产物 `session/GLM-5.3-Flash-Quark-Int8/<sid>/reports/final.{md,json}`
