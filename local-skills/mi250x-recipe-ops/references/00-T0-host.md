# T0 · 主机层（与任何版本无关）

> 本文件内容对**所有**引擎/ROCm/torch/模型/量化组合都成立。若你发现某条在这里但随版本变化，它放错了层——请移走而不是就地加注。

<!-- ── 搬运自 SKILL.md L31-66 ── -->
> **T0 · 主机层** — 机器身份 / 前置检查 / 起停观测 / 测量三道硬门。与引擎、ROCm、模型、量化版本**全部无关**。

## Host facts (never re-derive)

- Repo root: `/home/qiba/ai` (all paths below are relative to it).
- 8x MI250X GPU-dies (gfx90a). VRAM usage per die in **bytes**:
  `paste <(seq 1 8) <(for c in /sys/class/drm/card{1..8}; do cat $c/device/mem_info_vram_used; done)` — idle ≈ 1.0e7.
- Power cap 560 W/module (`/etc/amdgpu-powercap.conf`); align cap before any
  cross-run comparison.
- Port registry + mutual exclusion: `config/ports.conf`. Never assume a port
  is free; probe: `nc -z 127.0.0.1 $P; ss -ltnp | grep ":$P "`.

## Step 0: pre-flight before starting anything

1. Check which dies are occupied (command above).
2. Check who holds kfd: `sudo -n fuser -v /dev/kfd 2>/dev/null || fuser -v /dev/kfd`.
3. Check target port + its PID file: `cat logs/*-$P.pid 2>/dev/null`.
4. Align/inspect power cap if the comparison depends on it.

## Start / stop / observe

- Start = run the arm's `entry` script from `data/arms.json` (it bakes in all
  fail-closed guardrails; do NOT re-implement its args).
- Stop = kill the session group of the PID file only:
  `kill -TERM -$(cat logs/<key>-<port>.pid)`.
  **NEVER** `pkill -f` and never feed `pgrep` output to `kill` — this host runs
  many concurrent instances and has killed a production endpoint that way.
- Observe: `docs/日志规范-2026-09-12.md` conventions; logs + PID under `logs/`.

## Performance measurement rules (three hard gates)

1. Warm up first — the first request pays Triton autotune/JIT (26 vs 62 t/s on
   Ornith). Never report the first number.
2. Cross-boot drift sigma ≈ 4.6% — any claim below ±2% from a single boot is
  invalid; never mix absolute values across boots/builds in one table.
3. If single-stream TPS < 20, stop immediately, record "性能不达标，止于 X t/s",
   do not fill in remaining dimensions.


<!-- ── 搬运自 SKILL.md L320-339 ── -->
> **T0 · 主机层** — 起服硬门与单臂探针（脚本级，与版本无关）。

## 起服前的门 + 单臂探针（2026-09-21 新增，可直接复用）

- `quark-int8/gpu_gate.sh`：`source` 后 `gate 8121`（放行返回 0）/ `wait_free 8121 7200`。
  判据两条：除自己端口外无别的 `api_server` + 八张 GCD 各 <5 GiB（与 launcher 硬门同阈值）。
  内含三条单测与两个真实踩过的坑：`ps -eo args | grep -F '…api_server'` 会匹配 **grep 自己**
  ⇒ 门永远不过；`[ -ge $$… ]` 里的 `$$` 是 PID ⇒ 超时判断失效。
- `quark-int8/stack_probe.sh <臂名> KEY=VALUE …`：等卡 → 起服 → 事实召回 + TPS(conc 1/8/32)
  → 停服 → 追加 `logs/stack/results.jsonl`。fail-fast：每轮查容器 `State`，**连续两次**判死才撤，
  且**先把整份容器日志存盘再删容器**；`SKIP_BOOT=1` 可脱离 GPU 自测判据；
  `TPS_ISL_MULT=8/24` 切长上下文档（DCP/split-K 这类杠杆必须在长档判）。
- **一臂一杠杆**：混测只能探天花板，不能记收益。今天的教训：首轮五杠杆混测得 conc32 +47%，
  记在 split-K 头上；消融后真身是 `DSV41_IDX_AITER_KERNEL=1`（+69.3%），split-K 实际 −13%。
- 结论与数字出处：`hyperloom/reports/models/glm53-int4/decode-config-ablation.md`
  （机器可读同目录 `stack_results_20260921.jsonl`；`quark-int8/logs/` 是 gitignore 的）。
- 三个默认值即地雷，改 launcher 时别踩：① 枚举型 env 用 `${VAR:-}` 注入空串 ⇒ vLLM 抛
  `Invalid value ''`；② `MAX_CUDAGRAPH_CAPTURE_SIZE=0` + `ENFORCE_EAGER=0` ⇒ 断言拒绝（eager=0
  这条路此前从未走通）；③ `DSV41_IDX_AITER_KERNEL` 未设时走的是**本机不可信**的 torch 回退，
  而设 `=1` 实测 +69%。




---

## 附录 · 平台级结论（收编自 `docs/MI250X-会话经验总结-2026-09-03.md`、
## `docs/MI250X-GPU组conf与NUMA接线-2026-09-15.md`）

- 🔑 **`hipIpcOpenMemHandle` 在 gfx90a 必须 `flag=1`**（`LAZY_ENABLE_PEER_ACCESS`；
  `flag=0` → `hipErrorInvalidValue`）。这是 vLLM 自带 **CUSTOM allreduce 在 MI250 坏的直接根因候选**；
  跨进程远程 P2P **写**本身正常。〔§0.8/§5.1 L22-23, L114-126〕
- 🔑 **纯 GPU 排序（无 CPU 同步）在 gfx90a 不可达**（5 配方实证）：triton spin / HIP plain /
  HIP atomic release-acquire + `__threadfence_system` **全部首轮即错**（err 9.7~11.75）⇒
  **两个独立 HIP 进程的 IPC 内存不具备跨 GPU happens-before**，正确性必须由 **RCCL 级传输**担保。
  ⇒ one-shot 的 6–8 µs 目标**只在"不需要排序"时成立**。〔§5.1 L120-124〕
- **功率控制是模组级**：1 cap / 模组，2 die 共享，**副 die 读 0**。
  满载爬升瞬态**跳闸两次**（560 W 默认、400 W cap 各一次），**350 W cap 全程通过**。
  〔§0.6/§0.7 L20-21〕
- **GPU↔NUMA 映射表**：`GPU_DIE_NUMA_MAP` = `"<die>:<node>"`
  （`11/14/31/34 → node0`，`8e/93/ae/b3 → node1`）。
  **单 node 组（tp2/tp4）整进程一个 `NUMA_BIND` 前缀 ≡ 逐 rank 绑**；
  **tp8 的 rank 分属两个 node ⇒ numactl 单前缀必绑死一侧 ⇒ `NUMA_BIND=""`**
  （真 per-rank 绑定需引擎补丁）。〔NUMA 接线 §3 L36-40〕
- ⚠️ **`GPU_CPUFREQ_PERF=1` 从未上过机**：09-15 实测 48 个 cpufreq policy 全是 `schedutil`、
  `boost=1` ⇒ performance 档未验证；且重启失效（常驻需自装 systemd oneshot，**未装**）。
  **vLLM tp2/tp4 臂默认开 `NUMA_BIND` 是行为变化**（8113/8101/8301/8111 历史上都没绑过）
  ⇒ 建议先 ABAB 对拍（`NUMA_BIND=""` vs 默认）再决定是否常驻。〔§3 L41-44、§5 L62-66〕
- **ROCm 版本号三处易混**（都合法，别当成不一致）：`/opt/rocm/.info/version` 曾= **6.4.4**；
  amdgpu DKMS **6.16.13**（驱动系 30.30.04）；`rocm-smi-lib` 包版本 **7.7.0**。
  现役以 `/opt/rocm-7.2.4` 前缀为准。〔§0.2 L13-14〕
- **llama.cpp 构建身份自检两行**（版本无关的姿势）：
  `objdump -p bin/libggml-hip.so | grep -E 'NEEDED.*amdhip|RUNPATH'` ＋
  `strings bin/libllama.so | grep -c glm5next`（应 = 80）。
  🛑 **`.so.7`(7.2.4) 与 `.so.6`(6.4.4) 绝不进同一 `LD_LIBRARY_PATH`** ⇒ Bus error/segfault。
  另：`/usr/bin/hipcc` 默认指 6.4.4 而 `/opt/rocm` alternatives 指 7.2.4 ⇒ **路径全部写死**。
  〔GLM-5.3 全记录 §2.2 L51-56, L70-75〕
- **`-c` 是【总池】不是每槽**（`llama-context.cpp:291/293`，本 build `kv_unified='false'`
  ⇒ `n_ctx_seq = n_ctx / n_seq_max`）⇒ 「N 槽 × 每槽 262144」**必须写 `-c N*262144`**；
  按"每槽"理解去写 `-c 262144 -np 3`，**每槽会被悄悄切成 87381**。
  〔`docs/下载-ModelScope-Ornith-1.5-397B-Q8_0-与8110底座-2026-09-09.md` §14 L357-363〕
- **长跑下载的 `setsid` + 日志 + PID 必须与目标数据在同一存续域**：
  曾出现"日志/PID 落在会被别人摘掉的盘上 ⇒ 下载器被带走"。同文 §3 L60-74

---

## 附录 · 停服的**安全边界**（技能正文只给了句式，这里给"什么时候会反噬自己"）

> 权威源 `docs/recipes/ops/serve-start-stop-observe.md` §2 L44-48（2026-09-21 更新）。
> **这一节存在是因为 `kill -TERM -$(…)` 不是无条件安全的**——照抄会打死自己的 shell。

- ⚠️ **组杀的适用边界**：`nohup` 起的 server 若**继承了调用方的 PGID**，
  `kill -TERM -<pgid>` 会**连坐自己**（8110 扫描脚本实踩过）。
  ⇒ **只有确认 server 的 PGID 独立（`setsid` / 终端手起）才组杀，否则只杀该 PID。**
  手工起服请一律 `setsid`（只 `nohup` 时调用被打断会连带打死服务）。
  并发块另一处实录：**`kill -KILL -$pid` 已两次打死自己的 shell** ⇒ 停服走三步法，别直接 `-KILL` 组杀。
- 🛑 **`VLLM::Worker_TP<n>` 不在 PID 文件里，且新老长得一模一样**：EngineCore 会 fork 同名 worker，
  `ps` 里同一名字既可能是**别人刚起的服务**，也可能是**自己上一轮的死残留** ⇒
  只看名字 + 启动时长会**把 2 分钟前刚起的生产 worker 当成自己的残留杀掉**
  （09-21 实录：`Worker_TP5` 355027 启动 2:37，查父进程才发现其 `PPID` = 刚起的 `EngineCore` 354703）。
  ⇒ **杀任何 worker 前先核父进程链与启动时刻**：
  `ps -o pid,ppid,lstart,args -p <pid>` 再 `ps -o pid,lstart,args -p <ppid>`；
  **父进程是活的 EngineCore / `vllm serve` 就一律不动。**
- 🛑 **会话起的服务在 `dsh-web.service` 的 cgroup 里**（`setsid`/`nohup` 只改会话**不改 cgroup**）
  ⇒ 重启 DSH 面板服务曾**连坐杀掉在跑的 8112**（09-12 21:54）。
  已加 drop-in **`no-kill-descendants.conf`（`KillMode=process`）——别把它撤了**；
  且重启面板前后仍应按 PID 文件复核服务是否活着。
- ⚠️ **闸口那条「八张 GCD 各 <5 GiB」是"没有别人在跑"的判据，不是"能不能起"的判据**：
  vLLM 起服即按 `--gpu-memory-utilization` **一次性预分配**（09-21 实测：一个 TP8 服务在
  **权重才加载到 29%** 时，8 张 die 已各占 **52/64 GiB**）⇒ 卡上躺着 52 GiB **不代表它已就绪、
  更不代表可以被顶替**。判据错配的后果是**把自己的服务起在别人正在加载的服务旁边**。
- **显存观测的三条口径**（都错过）：
  ① 以 sysfs 为准且**单位是 bytes 不是 MiB**（`rocm-smi` 输出的才是 MiB）；
     按 MiB 文本解析恒得 0，曾把"等显存释放"变成空转；
     **`card0` 是 BMC 显卡（无 `mem_info_vram_total`）**，8 个 die 是 **card1–card8**，
     **HIP index i → card{i+1}**；
  ② **sysfs 快照不是驻留/进度的可靠判据**（缓冲一次性 alloc，瞬间跳满后拷贝期读数不动；
     09-04 见过加载中 8 die 全读 10 MiB 而 worker 自报 13.08 GiB）
     ⇒ 判驻留/OOM **只认 vLLM 自报的** `Model loading took X GiB memory` /
     `N GiB is allocated by PyTorch`；
  ③ **进程消失 ≠ 显存释放**（实测 kill 完 die 上还留 40+ GiB；8110 基脚本为此内置
     `VRAM_WAIT_FREE_S=150` 等待环）。
- **同一条臂可能有 3 份实体**（基脚本 / `launcher/favor/…_final_….sh` 薄封装 / `/mnt` 镜像）⇒
  **改动只落基脚本 + `md5sum` 对账 + 侧车软链**，否则会重演覆盖事故。
