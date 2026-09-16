# 模型无关的产出：编排缺陷、预算算术、环境/机架观测

这里的东西对**任何模型**都成立，不归属到具体模型。凡是从某个模型的现象推出的结论，都标了出处，并注明可否外推。

## 1. 三个上游 issue 的归因（哪个模型撞上的）

| Issue | 现场模型 | 性质 |
|---|---|---|
| [#1504](https://github.com/AMD-AGI/Hyperloom/issues/1504) enablement 补丁证明了却打不进去 → `enablement_stalled` 零基线 | **dense INT8**（`20260914T134902Z-244d9455`） | 编排/预算缺陷，模型无关 |
| [#1505](https://github.com/AMD-AGI/Hyperloom/issues/1505) 父进程 `HIP_VISIBLE_DEVICES` 泄进 ROCR 掩码子环境 | **dense INT8** 现场 | 环境掩码缺陷，模型无关 |
| [#1506](https://github.com/AMD-AGI/Hyperloom/issues/1506) 角色卡死在 `max_turns=12`；kill+resume 留下幻影租约 | **dense INT8** 现场 | 编排缺陷，模型无关 |

**重要更正**：#1505 与 #1506 此前被当成两件独立的事（一个掩码、一个租约），但它们确实是**同一套编排缺陷对启动失败的不同表现**——启动失败 → enablement 循环 → `max_turns`/租约问题。注意 **MoE session 的启动失败根因与掩码无关**（是 vLLM 的 MoE 后端选择），见 `../models/ornith-35b-a3b-moe/`。两个模型的失败**形态相同、根因不同**，这正是必须分账的理由。

`#1504` 的机械成因已定位：`orchestrator/actions/executors/_grid_runner.py:1403-1404` 写明每个 variant cap 都按 1024/1024 合成形状设定——**`integrate` 7800s**、`explore` 2400s、conc sweep 1800s。3h 预算下 `integrate` 占 72%，因此**只在 session 前约 50 分钟内可被调度**；而前 50 分钟被 PRELUDE 占满（`PRELUDE budget consumed 1129% (3656s of 324s cap)`，`--max-minutes-prelude-pct` 默认 0.03）。补丁"证明了却永远打不进去"是算术必然。可用纯 CLI 绕过（调大 `--max-hours`）。

## 2. accuracy gate 本身是预算大户（模型无关，但严重）

`RUN_EVAL` 默认 `"true"`（`actions/executors/_workload_envs.py:1741-1743`），会给**每个** baseline/candidate 追加一次 gsm8k `lm_eval`（`num_concurrent=64`，`max_tokens` 2048–4096）。在 dense INT8 session 上实测吃掉 120 分钟预算中的约 40 分钟：

- **约 7 分钟卡在第一个请求之前**：`datasets` 即使已完整缓存 `openai___gsm8k` 仍去重试 HuggingFace hub 元数据。受限网络下这看起来就是挂死——首个采样点：lm_eval 已 ~100% CPU 跑 90s 而 `POST /v1/chat/completions = 0`，socket 停在 `SYN_SENT:443`。它会自行恢复，但没有可预期的超时边界。
- 缓存里还留着上次崩溃的陈旧锁：`~/.cache/huggingface/datasets/_root_.cache_huggingface_datasets_openai___gsm8k_main_0.0.0_740312add….lock` 与同目录 `…_incomplete_info.lock`。
- 之后实测 **39.8–40.4 req/min**，1319 题约 33 分钟。

逃生口：`--no-eval` / `RUN_EVAL=false`（`cli/parser.py:716`；`actions/executors/explore.py:1274` 自身在某些决策上也会置 false）。但没有任何地方把这个取舍告诉操作者。

## 3. 容器内结束/清理不完整（与 #1506 Part B 同源）

- dense INT8 session 于 19:33 UTC 正常 CLOSE 后，**整簇 ray 仍在跑**（`gcs_server` / `raylet` / 48× `ray::IDLE`，挂了 10 小时）。
- `ray stop --force` 报 `Stopped only 0 out of 55 Ray processes within the grace period 16 seconds`，遗留 **49 个僵尸**。根因：**容器 PID 1 不回收孤儿**，而 orchestrator 的父进程是 `docker exec`（父死后子进程被 reparent 到 PID 1，无人 reap）。
- 这直接解释了 `--resume-from` 里 "dead holders" 的来源：上一个进程的子进程全成了僵尸，`dispatcher: reclaimed 1 running task(s) with dead holders` + `resource_lock: reaped 5 lease(s) ... (pid=<上一个 resume 的 pid>)` 确实会跑，但**在首个 tick 之后**，而 `enablement_round_in_flight` 这个 guard 并不随 holder 失效，于是仍然拦 `baseline`。缺陷是 "guard 未随 holder 失效"，不是 "回收器不跑"。
- 可靠清法：`docker stop <container>`（杀 PID 1 → 全部回收）。

## 4. 机架观测陷阱：宿主 `rocm-smi` 看不到容器的 GPU 活动（**已证伪一条 scout 结论**）

某会话的 research scout 以 `sev=high` 报过"硬件钉频"：

> sclk 在 3766s baseline 的全部 1748 个采样里恒定 800MHz / fclk 400MHz（MI250X boost 应约 1.7GHz/300W，而 baseline 只吃 96–98W 平线）……每个 compute/issue-bound kernel 白付 ~2.1x，任何配置或补丁杠杆都救不回来。

**实测证伪**：容器内在负载下 **sclk = 1565MHz**，boost 正常。陷阱在于**宿主侧 `rocm-smi` 看不到容器的 GPU 活动**——一个已确认在跑的 fp16 GEMM 循环（容器内 `206% CPU`）期间，宿主报的是 `GPU use 0%` / `Average Graphics Package Power 96-97W` / `sclk 800MHz`，即**空闲画面**，8 个 GCD 全程如此。

所以：任何从容器边界错误一侧采样时钟的 agent 都会得出"整机被限频"并放弃寻找真杠杆。**读时钟必须用启动 server 的那个 namespace。**

## 5. 一条已被推翻的旧结论

早期我把 AITER 的 **ASM kernel 普查结果**当成了运行时判决，写下"gfx90a 上 MoE 路径死（缺 bf16 原子）"。这是**混淆了两件事**：

- 普查说的是**预编译码对象的二进制补丁可移植性**：`port-matrix.md` 统计 1422 个 gfx942 kernel，242 个可通过二进制补丁重编码到 gfx90a，539 个因 `global_atomic_pk_add_bf16` 被挡，647 个因无 FP8 ALU 被挡。文档自己写明这 1180 个"并不代表失去的能力"，且 242 是**二进制补丁**的上限，不是源码级移植或 Triton 的上限。
- 运行时**从未**因为原子指令失败过。MoE 模型的实际失败是 vLLM 的 MoE 后端选择——见 `../models/ornith-35b-a3b-moe/`。

同一逻辑的现成反证就是 dense INT8 模型：AITER 那 482 个 int8 ASM kernel 同样被判"不可移植"，而该模型跑到了 444.9 tok/s，只因为 vLLM 走了非 ASM 路径。

## 6. 通用方法论教训（写给后续 agent/操作者）

1. **基线先于优化**：只要第一个 baseline 起不来，enablement 循环会吃掉整场 session，且失败形态（`enablement_stalled`、0 baseline）会掩盖真正的启动原因。
2. **预算要对着 cap 表算**，别对着"感觉"：`integrate` 的 cap 是 7800s，任何短于 2.2h 的预算都注定无法落地补丁。
3. **精度门要显式决策**：`RUN_EVAL` 默认开，代价约 1/3 预算。
4. **观测要在正确的 namespace 里做**（容器内外 `rocm-smi` 不等价）。
5. **同一类失败会在不同模型上重复出现**——但根因可能完全不同。这正是本仓库按模型分账的理由。
