---
name: mi250x-recipe-ops
description: >-
  Operates the local 8x AMD MI250X (gfx90a / MI200) workstation using its
  verified recipe library: which serving arm runs on which port, how to
  start/stop/observe it, how to rebuild its environments (vLLM 0.28 / master /
  nightly-0918 image, llama.cpp gfx90a, ROCm 7.2.4, wu1w INT8), how to
  apply/verify/revert its patches (QuickReduce, splitKV, DSA indexer, AIS/PLE,
  aiter), how to set its tuning knobs (MTP/spec depth, AIS, --fit off, TP
  split-mode), and which routes are proven dead and must not be re-attempted.
  Every piece of knowledge is tagged with its applicability scope across nine
  axes (host / driver / ROCm / PyTorch / engine / model architecture / weights /
  quantization / topology) so conclusions from one engine version, ROCm
  version, model or quantization are never misapplied to another. Use when the
  user asks about "this machine's" models/ports/launchers/recipes, mentions
  ports 8101..8127/8203/8301/8302, asks to start/stop a local endpoint on
  MI250X hardware, asks whether some finding "still applies" after changing
  engine / ROCm / torch / model / quantization, or wants Hyperloom's local
  RecipeKB warm-start for these arms. Do not use for MI300X+ or Docker-based
  generic serving (see serving-llms-on-instinct).
allowed-tools: Bash, Read
---

# MI250X Workstation Recipe Ops

单机操作手册：`/home/qiba/ai`（下称 `$AI`）＝ 8 × AMD MI250X（gfx90a），宿主 ROCm 7.2.4，
vLLM（0.28.0 / master / nightly-0918 镜像）与 llama.cpp 两支引擎，每条服务臂一份 launcher。

**权威源**：配方 markdown 在 `$AI/docs/recipes/**/*.md`（闸口 `$AI/tools/audit_recipes.py`）。
`data/*.json` 是给 agent 的机器可读抽取；**与配方 markdown 或 launcher 冲突时以它们为准**。

---

## 0. 怎么读这份技能：适用范围分层

**这是本技能的第一道纪律。** 一条在 vLLM 0.28.0 + bf16 + TP8 上测出来的结论，被读到
llama.cpp Q4_K_M 臂上引用，就会得出错误动作。所以每条知识都带有适用域；
**你要做的动作与适用域任一轴不符时，该结论不适用——不是"参考一下"，是不适用。**

### 0.1 九轴

| 轴 | 问什么 | 本机取值（全表见 `data/scope.json`） |
|---|---|---|
| `host` | 哪台机器 | `mi250x-gfx90a`（唯一） |
| `driver` | 内核驱动 / 固件 / 功率 | `amdgpu-dkms-6.16.13.30300400`、`power-cap-560w` |
| `rocm` | **哪一套** ROCm（宿主 / 容器内 / wheel 标签是三个东西） | `host-7.2.4`、`container-7.2.3-nightly0918`、`wheel-tag-rocm723`、`legacy-6.4.4`、`10.0.0`(已移除) |
| `torch` | 哪个 PyTorch（**同机并存三个**） | `2.12.0+git6bbd260`、`2.14.0+rocm7.2`、`2.11.0+rocm7.2` |
| `engine` | 哪个引擎 + 哪个构建 | `vllm-0.28.0+rocm723`、`vllm-master@8a728663c`、`vllm-0.3.1.dev85+gdee37d891`、`llamacpp-0.3.0-dev@a9e9c3c5f` / `@629b50552`、`llamacpp-0.4.0-dev@f37da57`、`sglang-not-deployed`、`atom-vllm`(判死) |
| `arch` | 模型架构（**≠ 模型名**；决定 arch 注册表命中） | `glm_moe_dsa`、`glm5next`、`deepseek_v32`、`deepseek_v41`、`deepseek_v4_flash`、`qwen3_5`、`qwen3_5_moe`、`qwen4exp`、`qwen2`、`mamba-gdn-hybrid`、`mla` |
| `model` | 哪一份权重（**开集**；换出品方、或本机自转都算换底座） | HF id 或 `models/` 下路径 |
| `quant` | 哪种量化（决定内核路径） | `bf16`、`fp8`、`fp8-emulation`、`int8-w8a8`、`mxfp4-w4a16`、`ct-int4-w4a16`、`gguf-{q8_0,q4_k_m,ud-q4_k_xl,ud-q8_k_xl}` |
| `topology` | 并行度与占卡 | `tp1/tp2/tp4/tp8`、`dcp1/dcp8`、`gpu-group` |

### 0.2 四个分层

| 层 | 是什么 | 失效于 |
|---|---|---|
| **T0** 通用 | 硅、驱动、文件布局、运维纪律。与引擎/框架/模型/量化版本**全无关** | 换机器、换驱动、换端口规划 |
| **T1** 版本 | 绑定 engine / rocm / torch 某个具体构建 | 升级或降级任一轴 |
| **T2** 模型 | 绑定权重身份与量化形态 | 换 checkpoint / 出品方 / 量化方案或 group size |
| **T3** 组合 | 绑定完整组合，通常是一条服务臂或一次实测 | 任一轴变化；**且跨 boot 绝对值不可混** |

### 0.3 匹配用 `applies_to`，不要用 `consumed_by`

三个字段长得很像，语义完全不同：

| 字段 | 含义 | 回答的问题 |
|---|---|---|
| `applies_to` | **适用判定**：九轴谓词 | 「换到我的目标上还成立吗」 ← **匹配用这个** |
| `consumed_by` | **溯源/历史**：真正消费过它的臂 id | 「这东西在野外哪里被跑过」 |
| `effect_by_scope` | **分档效果**：在某个 scope 下是正是负 | 「适用 ≠ 有收益」 |

⚠️ **实测教训（2026-09-21）**：`consumed_by`（旧名 `scope`）**不能**用来反查适用性。
证据：`8121` 被 **0 个 knob** 认领，但它实际适用其中 6 个（含 `quick-reduce-gfx90a`、
`vllm-fused-moe-tile-seeds`——这两条边只存在于 8121 自己的 `related:` 里，是**单向**的，
全库 65 条 knob→臂边里 63 条这样单向）。该字段还混过 knob id、散文与**否定**消费者
（`"8111-…（不接投机，仅为记录）"`）。**它是阅读日志，不是规则。**

**怎么用**：

```bash
# 以某条现成臂为目标，问"适用什么"
python3 /home/qiba/ROCm.AI/scripts/scope_match.py --arm 8121-glm-5.3-ct-int4-w4a16-vllm-tp8
# 直接给轴（给的越多判定越准；未给的轴 = unknown，不否决但判定不完整）
python3 /home/qiba/ROCm.AI/scripts/scope_match.py --engine vllm-0.28.0+rocm723 --quant int8-w8a8 --topology tp1
# 双向差集闸口：适用却无人记录 vs 记录了却判不适用（后者是危险矛盾）
python3 /home/qiba/ROCm.AI/scripts/scope_match.py --orphans
python3 /home/qiba/ROCm.AI/scripts/scope_match.py --axes      # 打印九轴词表
```

轴的三态必须分清：**具体值** = 有限定；**`"*"`** = 明确任意；**缺失** = 未知（不否决）。
glob（`vllm-*`）是必需的——跨臂开关的**规则**对任何 vLLM 版本都成立，差别在**效果**。

⚠️ **两条使用注意**（都是实测踩出来的）：
- **只给三两个轴会得到「0 条适用 / N 条判定不完整」——那不是"没有可用知识"，是轴不够。**
  要精确判定就用 `--arm <id>`（它自带全部九轴），或把 `--host/--driver/--rocm/--torch` 也补上。
- **退出码与判定结果无关**：`0` = 查询跑完（哪怕全是"不适用"），`2` = 输入错误，
  `1` 只用于 `--orphans` 发现**矛盾**。**答案看输出，别拿退出码当判据。**

### 0.4 引用纪律

- 数字只有 **scope 完全一致 + 同 boot + 同 power cap** 才可横比。
- 跨 boot 漂移 σ≈4.6% ⇒ ±2% 级差异不得当结论。
- **一个修复用两把尺子测出的两个增益不可相加**（同一件事的两次测量）。
- 单流 TPS **必须带 ctx**；TPS 测量必须 `ignore_eos=True` + `min_tokens=max_tokens`，
  否则短答立刻 EOS、解码 TPS 变 nan。
- 只在 int4 上验过的结论不得外推到 bf16 或 int8——**同一模型换量化，
  失败现象可能一样而根因完全不同**。
- **版本绑定必须逐文件核，不能按文件名推**。实测：某转换记录全文只有 nightly 一条线
  （`0.28.0`/`master`/`748B`/`GLM-5.3` 全 0 命中）；另一份内核复核**全文无「ROCm 7.2.4」字样**。

### 0.5 references/ 路由表

SKILL.md 是常驻的索引与 T0 层；下列细节**按需加载**：

| 文件 | 装什么 | 什么时候读 |
|---|---|---|
| `references/00-T0-host.md` | 主机层全文（含起服门、单臂探针、测量方法） | 起服/测量/判死之前 |
| `references/10-version-matrix.md` | 版本层（aiter JIT 复用、官方技能仲裁、DCP/fp8KV、补丁队列、镜像坑） | 换引擎/ROCm/torch，或跨版本搬结论 |
| `references/20-model-quant-matrix.md` | 模型×量化（GLM-5.3 自转 int4、Ornith 四线转换规程） | 碰自转权重或换量化 |
| `references/30-arms-detail.md` | **逐臂 T3 细节**（同一臂不同时点的数字口径、3 份实体分叉、日志侧车、互斥实况） | 搬某臂的经验到别处之前 |
| `references/40-knobs-and-patches.md` | 自研 kernel（MoE GEMV / 稀疏 split-K）、roofline 口径、**护栏全谱与桩干跑**、`block_size` 才是真锁 | 调性能杠杆 |
| `references/50-dead-routes.md` | 死路总表（含 2026-09-21 新增 17 条判负路线与"别搬"清单） | **动手前必读**，防止重做 |
| `references/60-method-measurement-gates.md` | **测量有效性、判据设计、闸口自身缺陷、跨模型外推禁令、制度八条** | 设计任何实验/判据之前 |
| `data/scope.json` | 九轴词表 + 分层规范 + **8 条已知混淆点** | 任何"这版本到底是多少"的疑问 |
| `data/*.json` | 逐条目的 `applies_to` / `effect_by_scope` / 实测数字 | 用 `scope_match.py` 或直接查 |

---

## 1. T0 · 主机层（与任何版本无关）

### 1.1 机器身份（不要重新推导）

- 仓库根：`/home/qiba/ai`（下列路径均相对它）。
- **8 个 GCD，每个 64 GiB HBM / 104 CU**。**两个 GCD 才是一个 MI250 模块的 128 GiB** ——
  算 KV 时绝不能按 128 GiB/GCD 算。PCI ID `1002:740c`。
- **无原生 FP8 / FP4 矩阵核**（CDNA2）。int8 MFMA 有，bf16 MFMA 有。
- PCIe Gen4 x16（`current_link_speed` 16.0 GT/s）。
- CPU 2 × EPYC 7413（96 线程），NUMA 2 节点；**GPU NUMA 接线 4+4**：
  `card1-4 → numa_node 0`，`card5-8 → numa_node 1` ⇒ TP8 跨 NUMA。
- 驱动：**`amdgpu-dkms 1:6.16.13.30300400` 装在内核 6.8.0-139-generic 上**。
  ⚠️ `uname -r` 给 6.8.0-139，`/sys/module/amdgpu/version` 给 6.16.13，**两者不等是正常的**
  （DKMS 回移，vermagic 仍是 6.8.0-139）。别把 6.16.13 当内核版本。
- 固件包 `amdgpu-dkms-firmware 30.30.4.0.30300400`。
- ⚠️ **`card0` 是 BMC 显卡（无 `mem_info_vram_total`）**，8 个 die 是 **card1–card8**，且 **HIP index `i` → `card{i+1}`** ⇒ 拿 `card{0..7}` 循环会读到一个不存在的 die、同时漏掉一个真 die。
- 功率：`560 W/module`，`/etc/amdgpu-powercap.conf`（`POWER_CAP_UW=560000000`）。
  **跨 run 比较前必须先对齐 cap**，报告数字必须带 cap 口径。
- 显存读数单位是 **bytes**（`rocm-smi` 输出的才是 MiB）：
  `paste <(seq 1 8) <(for c in /sys/class/drm/card{1..8}; do cat $c/device/mem_info_vram_used; done)` — 空闲 ≈ 1.0e7
  ⚠️ **按 MiB 文本解析 `mem_info_vram_used` 会恒得 0** ⇒ "等显存释放"变成空转；且 **sysfs 快照不是驻留/进度的判据**（缓冲一次性 alloc，瞬间跳满后拷贝期读数不动；09-04 见过加载中 8 die 全读 10 MiB 而 worker 自报 13.08 GiB）⇒ 判驻留/OOM 只认 vLLM 自报的 `Model loading took X GiB memory`。**进程消失 ≠ 显存释放**（实测 kill 完还留 40+ GiB）。详见 `references/00-T0-host.md` 附录。

### 1.2 安全红线（每条都有真实事故）

- 🛑 **TP8 全 die 同步计算曾三次整机断电**（`config/gpu_mi250x_tp8.conf:49`）⇒
  **本组脚本对功率 cap 只读不改**。改 cap 前先想清楚这一点。
- 🛑 **绝不抢卡、绝不杀别人起的服务**。任何占用 GPU 的运行（起服、微基准、profiler）
  **先取得用户明确许可**；需要 GPU 的结论一律先提方案、等批准。纯 CPU 工作不受此限。
- 🛑 **停服只认自己的 PID 文件 / 容器名**：`kill -TERM -$(cat logs/<key>-<port>.pid)`。
  **禁止** `pkill -f`、**禁止**把 `pgrep` 输出喂 `kill`、**禁止** `ps | grep | kill`。
  本机实录两类事故：① `pkill -f` 命中自己这条命令行（把当次调用杀掉 3 次）；
  ② `pgrep -x llama-server` 同时命中探针与生产实例，**误杀 8108**。
  ⚠️ 例外：`8121` 的 PID 文件里存的是**容器 ID** ⇒ 该臂只能 `docker rm -f $(cat …)`。
  ⚠️ **`kill -TERM -<pgid>` 不是无条件安全的**：`nohup` 起的 server 若**继承了调用方的 PGID**，组杀会**连坐自己的 shell**（8110 实踩过；`kill -KILL -$pid` 已两次打死自己的 shell）。**只有确认 server 的 PGID 独立（`setsid` / 终端手起）才组杀，否则只杀该 PID**；手工起服一律 `setsid`。
  🛑 **`VLLM::Worker_TP<n>` 不在 PID 文件里且新老同名** ⇒ 杀任何 worker 前先核父进程链（`ps -o pid,ppid,lstart,args -p <pid>`）；**父进程是活的 EngineCore 就一律不动**（09-21 实录：曾把别人 2 分钟前刚起的生产 worker 当残留）。
  🛑 会话起的服务**在 `dsh-web.service` 的 cgroup 里**（`setsid` 不改 cgroup）⇒ 已加 drop-in **`no-kill-descendants.conf`（`KillMode=process`），别撤**。
  完整边界见 `references/00-T0-host.md` 附录。
- 🛑 **不碰 `/opt/rocm` 的 alternatives**。宿主 `/opt/rocm-7.2.4` 是**软链指向本盘解包树**
  （非 apt 完整安装）——**apt 会透过软链覆盖解包树，直接打挂正在跑的 `vllm_0.28.0_rocm72`**。
- 🛑 **`HIP_VISIBLE_DEVICES` 会泄进 ROCR 掩码子环境**，导致 `import vllm` 直接抛。
- **public 仓库**：不写 `git add hyperloom/session`；`session/**/runtime/` 一律不入库
  （`runtime/kernel-agent.env.sh` 是 0644 内含 API key 明文）。
- **只在需要时起服**；测量/验证一跑完立即停服，不留常驻服务占着 8 张卡。

### 1.3 Step 0：起任何东西之前

1. 哪些 die 被占（§1.1 的显存命令）。
   ⚠️ **「八张 GCD 各 <5 GiB」是"没有别人在跑"的判据，不是"能不能起"的判据**：vLLM 按 `--gpu-memory-utilization` **一次性预分配**——实测一个 TP8 服务在**权重才加载到 29%** 时 8 张 die 已各占 **52/64 GiB**。
   ⇒ 卡上躺着 52 GiB **不代表它已就绪、更不代表可以被顶替**；误读这条会**把自己的服务起在别人正在加载的服务旁边**。
2. 谁握着 kfd：`sudo -n fuser -v /dev/kfd 2>/dev/null || fuser -v /dev/kfd`。
3. 目标端口与它的 PID 文件：`cat logs/*-$P.pid 2>/dev/null`；**永不假设端口空闲**：
   `nc -z 127.0.0.1 $P; ss -ltnp | grep ":$P "`。
4. 端口登记与互斥：`config/ports.conf`（**端口权威**，登记到 8127）。
5. 需要对齐功率时看 cap。

**标准件**（别手搓）：`ROCm.AI/quark-int8/gpu_gate.sh`（`source` 后 `gate <port>` /
`wait_free <port> <秒>`；判据 = 除自己端口外无别的 `api_server` + 八张 GCD 各 <5 GiB）、
`ROCm.AI/quark-int8/stack_probe.sh <臂名> KEY=VALUE …`（等卡→起服→事实召回+TPS→停服→追加
`logs/stack/results.jsonl`；`SKIP_BOOT=1` 可脱离 GPU 自测判据）。
两条真实踩过的坑：`ps -eo args | grep -F '…api_server'` 会匹配 **grep 自己** ⇒ 门永远不过；
`[ -ge $$… ]` 里的 `$$` 是 PID ⇒ 超时判断失效。

### 1.4 起停与观测

- **起** = 跑该臂 `data/arms.json` 里的 `entry`（脚本自带 fail-closed 硬门，**不要重实现它的参数**）。
- **停** = 只杀 PID 文件里的进程组（见 §1.2）。停完**删把手**——死把手 = PID 复用后误杀。
- **看** = `docs/日志规范-2026-09-12.md` 的约定；日志与 PID 落 `logs/`。
- 闸口（只读，exit 1 = 红）：
  `python3 $AI/tools/audit_recipes.py`、`audit_ports.py`、`audit_conf_wiring.py`、
  `audit_log_paths.py`、`audit_ctx.py`、`audit_kill_discipline.py`、
  `bash $AI/tools/verify_env_patches.sh`、`bash $AI/tools/instruction_budget_check.sh`。
  ⚠️ **闸口全绿 ≠ 没有红线级问题**：`audit_kill_discipline.py` 的存在理由就是
  「其余四个 audit 全绿却与两个 P0 并存」。且 `audit_log_paths.py:86` 的正则
  **只匹配带引号的赋值** ⇒ 8109 臂 4 处 `>/tmp/` 漏检却报绿。

### 1.5 性能测量（三道硬门 + 两个测量有效性前提）

1. **先预热**：首个请求要付 Triton autotune / JIT（Ornith 上 26 vs 62 t/s）。**永不报第一个数**。
2. **跨 boot 漂移 σ≈4.6%**：单次 boot 上 ±2% 级的差距无效；
   **绝不把不同 boot/构建的绝对值混进同一张表**。
   ⚠️ **但这条有个更主要的方差来源被长期误归因**（`quark-int8/RESULT.md` §7ter 实测）：
   臂内单流离散 **12–24%**（n=4 11.8% / n=5 16.7% / n=6 24.0%）**几乎全是内容变化，
   不是计时噪声**——改成**固定 prompt + 丢弃首次请求**后离散降到 **0.1%**。
   ⇒ **项目的 1.036× 噪声门只在"固定 prompt"口径下成立**；
   用"加盐中位数"最多只能定到 ±10%，**检不出 5% 级效应**。
   所以单流数字必须带**负载标签**：同一臂同一配置，
   结构化可预测负载 ~86 t/s、加盐混合 ~68.83、开放分析型 ~56。
3. **单流 TPS < 20 就立即停**，记 `性能不达标，止于 X t/s`，不要填满其余维度。

测量**有效性**前提（两条，漏了会得到假结论）：

- **沙箱策略会影响测量**：`workspace-write` 沙箱下 **`/dev/shm` 不可写** ⇒
  SE-Bench Phase1 曾得出 **12/17 而非 17/17** 的**假 FAIL**。
- **指令预算 65536 会静默降级**：超预算时 **`AGENTS.md` 整份不进上下文**、`CLAUDE.md` 从尾部截断，
  模型只看到一行 marker ⇒ AGENTS.md 里的全局硬门（工具调用纪律、单流 TPS<20 立停）
  对所有会话形同不存在。**加指令内容前必跑** `bash $AI/tools/instruction_budget_check.sh`。
- **计时仪器本身有地板**：`rocprofv3` 在本机的中位 ≈ **4.96 µs**，该桶合计被系统性抬高
  ⇒ 会污染一切 profile 归因。用 profile 排名前先扣这个地板。

---

## 2. T1 · 版本层（速查；细节见 `references/10-version-matrix.md`）

### 2.1 版本指纹（`config/env-lock/*.txt` **不含 ROCm 版本**，必须三方拼）

| env | 引擎 | PyTorch | aiter | ROCm |
|---|---|---|---|---|
| `vllm_0.28.0_rocm72` | `vllm==0.28.0+rocm723` | 2.12.0+git6bbd260 | 0.1.19 | host 7.2.4 |
| `wu1w-int8-028` | 同上（+7 处 aiter/int8 手改） | 同上 | 0.1.19 | host 7.2.4 |
| `vllm_master_rocm724` | `0.1.dev1+g8a728663c.rocm724`（editable） | 同上 | 0.1.19 | host 7.2.4 |
| `pytorch_2.14_rocm7.2` | 无 vllm | **2.14.0+rocm7.2** | 无 | 7.2 |
| `unsloth_2026.9.2_rocm7.2` | 无 vllm | **2.11.0+rocm7.2** | 无 | 7.2 |
| 8121 臂（容器） | `0.3.1.dev85+gdee37d891` | 2.12.0+git6bbd260 | （关） | **容器内 7.2.3** |
| `wu1w-fresh` | `vllm==0.28.0+rocm723` | **未记录** | **未记录** | 未记录 |

`vllm_0.28.0_rocm72` 与 `wu1w-int8-028` 的 env-lock **`diff` 为空（rc=0）**；
`wu1w-fresh` 的 lockfile **全文件 1 行，是 stub，不能当证据**。

### 2.2 八个已知混淆点（全文见 `data/scope.json > known_confusions`）

1. `+rocm723` 是 **wheel 构建标签**，不是运行时 ROCm；wheel RUNPATH 写死的
   `/opt/rocm-7.2.3/lib` **本机不存在**，全靠裸 `/opt/rocm/lib` 兜住。
2. nightly-0918 **容器内 ROCm 7.2.3 ≠ 宿主 7.2.4**；容器不吃宿主 `/opt/rocm`。
3. `uname -r`(6.8.0-139) ≠ `/sys/module/amdgpu/version`(6.16.13)。
4. `data/arms.json` 的 `framework_version` 混了三种格式（发行版号 / master dev 号 /
   llama.cpp build）⇒ **跨引擎、跨格式都不可比**。
5. **同机并存三个 torch**。
6. `device_name`（MoE 调优表名）**不是本机常量**，随镜像/构建变。同一台机器实测出现过
   `AMD_Instinct_MI250X_MI250` 与 `AMD_INSTINCT_MI250_(MCM)_OAM_AC_MBA` 两个值。
7. 技能臂表 ≠ 全部在册臂：**以 `config/ports.conf` + launcher 实物为准**。
8. 两臂都写 "INT8" 不等于同一底座（8115 Quark INT8 专家 386 G vs 8116 CT-Int4 专家 193 G）。

### 2.3 与 `serving-llms-on-instinct`（AMD 官方技能）的仲裁

该技能直接读 `data/gpu_overrides.json > gpu_configs`（按 **gfx 架构**分档），
而它只覆盖 `gfx942`/`gfx950`——全技能内 MI250/gfx90a 命中 **0** 次。
**本机任务一律以本技能为准**，冲突处按下表：

| 通用口径可能怎么说 | 本机实测事实 |
|---|---|
| 启用 AITER 加速 | **不可用**（gfx90a 无 AITER MoE 路径） |
| 打开 QuickReduce | **看 KV 预算**：`init_custom_qr` 固定吃 ~9 GiB/卡；本模型 KV 只有 8.17 GiB 时**起不来** |
| 用 split-KV 提速注意力 | **默认 0**：conc 1/8/32 三档全负（−18%/−14%/−13%） |
| 关掉 eager 换 CUDA graph | 需同时给 `MAX_CUDAGRAPH_CAPTURE_SIZE>=1`，否则断言拒绝 |
| indexer 走框架默认 | **必须 `DSV41_IDX_AITER_KERNEL=1`**：默认是更慢且本机不可信的回退，开启 +69.3%（conc32） |

桥接（幂等，bootstrap 还原第三方文件后要重跑）：
`python3 hyperloom/patches-local/apply_serving_skill_mi250x.py`（`--check` 只看状态，`--revert` 回滚）。

### 2.4 SKU 级而非架构级

64 GiB/GCD、104 CU/GCD、TP8 时权重 52.9 GiB/rank ⇒ 32k 档 KV 仅 8.17 GiB / 94,016 tokens。
**别按「MI250X = 128 GiB」算 KV**（那是两个 GCD 之和）。
（2026-09-21 补正：52.9 GiB/rank 与 5.89 GiB KV 同属 **DCP=8** 那一行；
**DCP=1** 是 **50.84 GiB/rank + 8.17 GiB**。）

---

## 3. T2 · 模型 × 量化层（速查；细节见 `references/20-model-quant-matrix.md`）

**量化形态决定内核路径，因此决定一切。**

| 量化 | 本机可行性 | 关键事实 |
|---|---|---|
| `bf16` | ✅ 最稳 | vLLM 原生路径 |
| `fp8` | ❌ **硅门** | `torch._scaled_mm` 要求 MI300+/CC≥8.9；Ornith-1.5-397B-FP8 全量 389.6 GiB 权重装载健康后仍在 engine init die。官方 vLLM recipe 的 AMD 支持矩阵只有 {MI300X,MI325X,MI355X}，MI250X 全矩阵缺席 —— **这是硅门，不是调参问题** |
| `fp8-emulation` | ⚠️ 唯一路线 | `hyperloom/patches/fp8-w8a8-emulation-gfx90a/`；gfx90a 上唯一把这颗 FP8 checkpoint 装进 vLLM 的路线。代价：**BF16 权重常驻 48.27 GiB/die** ⇒ 1M 上下文装不下 |
| `int8-w8a8` | ✅ | 走 aiter `module_gemm_a8w8`（CDNA2 有原生 int8 MFMA）⇒ **本机唯一 AITER GEMM 可行路线**。默认只量化 routed experts：把 attn/GDN 也拉进 INT8 实测 **−8.5% 吞吐、NLL +2.8%（不加分）** |
| `mxfp4-w4a16` | ⚠️ **机制可达、性能未测**（见 §5 更正 1） | ROCm 上 MXFP4 MoE 只有 `AITER_MXFP4_BF16`（CDNA2 门控禁用）+ `EMULATION`，**无 triton 回落** ⇒ 本机实测逐 token 反量化、慢 5×。**省显存不省算力** |
| `ct-int4-w4a16` | ✅ | **int4 只能走 compressed-tensors**：Quark 对 int4/uint4 一律 `NotImplementedError: No quark compatible scheme was found`。走 MFMA（实测生成 `v_mfma_f32_16x16x16bf16_1k`、**零 FMA**） |
| `gguf-*` | ✅ | llama.cpp；**要单流速度就走这条路** |

**跨模型/跨量化禁令**：dense INT8 与 MoE bf16 的失败长得一样，**根因完全不同**。
一项结论只在某量化上验过，不得外推。

---

## 4. T3 · 服务臂（速查；全量见 `data/arms.json`）

✅ 本节现与 `config/ports.conf` 对齐：**8115/8116/8117/8119 四条臂已于 2026-09-21 补齐 serving 配方**（`$AI/docs/recipes/serving/`），并合成进 `data/arms.json`（19 条，全部带 `applies_to`）。端口权威仍是 `ports.conf` + launcher 实物。
⚠️ **臂表里的数字一律先核三轴**（引擎 × MTP/并行深度 × 负载标签）——本轮就是靠这条把「8116 的 38.17 vs 68.83 矛盾」判成**跨轴误标**而非真矛盾。

| 端口 | 臂 | 引擎 | 状态 |
|---|---|---|---|
| 8101 | Qwen3.8-27B Fable-Distill BF16 TP2 | vllm master | active |
| 8103 | DeepSeek-V4-Flash-0731 284B Q8_K_XL | llama.cpp | active |
| 8107 | Qwen3.8-Flash-Next 176B BF16 TP8 | vllm master | active |
| 8107 | Qwen3.8-Flash-Next 176B Q4_K_XL | llama.cpp | active（与上互斥） |
| 8108 | GLM-5.3-Flash 320B Q8_0 | llama.cpp | active |
| 8109 | 176B BF16 splitKV | vllm master | experimental |
| 8110 | Ornith-1.5-397B Q8_0 | llama.cpp | active |
| 8111 | Ornith-1.5-35B-A3B BF16 TP4 | vllm 0.28 | active |
| 8112 | DeepSeek-V4.1-Flash 748B Q4_K_M（常驻） | llama.cpp | active |
| 8113 | Qwen3.8-27B BF16 TP2 | vllm 0.28 | active |
| 8114 | Qwen3.8-27B INT8-W8A8 TP1 + dflash12 | vllm 0.28 | active |
| **8115** | Ornith-1.5-397B **Quark INT8 W8A8（Attn 变体）**（出厂/保守臂；⚠️ 文件名说 int8w8a8，默认权重是 `-Attn`） | vllm 0.28 | active |
| **8116** | Ornith-1.5-397B **CT-Int4 W4A16** + MTP(5)（单流最快 vLLM 形态 68.83 加盐；⚠️ 文件名写 mtp1，默认 `SPEC=5`） | vllm 0.28 | active |
| **8116** | Ornith-1.5-397B **CT-Int4 W4A16** + MTP | vllm 0.28 | active |
| **8117** | 同 8115 checkpoint + 三个杠杆（AITER int8 线性 / SPEC=5 / `--no-enable-prefix-caching`）。🛑 **做实验用 8127 不要用 8117** | vllm 0.28 | active |
| **8119** | DeepSeek-V4.1 CT-Int4 + **engram-int4**（与 8121 共用补丁树） | vllm nightly-0918 | **🛑 experimental — 默认权重是悬空软链，当前起不来**（`:183` fail-closed 会拒） |
| 8121 | GLM-5.3-CT-Int4-W4A16 402 GB（**本机自转**）TP8 | vllm nightly-0918 | active |
| **8127** | **实验臂专用端口**（09-18 两会话撞 8117、PID 文件互相覆盖后定的规矩） | — | — |
| 8203 | Qwen2.5-1.5B smoke | vllm 0.28 | active |
| 8301/8302 | 27B Fable splitKV / dflash2 | vllm 0.28 | experimental |
| ~~8207~~ | ROCm 10.0.0 A/B 臂的**墓碑，勿复用** | — | 已退役 |

**选型**：单流最快 → 8107 llama.cpp Q4_K_XL；OpenAI API + 大 KV 池 → 8107 vLLM BF16；
唯一常驻大模型 → 8112。完整理由见 `$AI/docs/recipes/README.md` §5。

🔑 **`8116` 的"38.17 vs 68.83 矛盾"已裁决（2026-09-21）：不是矛盾，是两个轴同时变了却没记。**
按 `quark-int8/RESULT.md` §7 的表格，`38.17` 属 **`nightly docker 0.29.1rc1` + MTP(1)**，
而 `68.83` 属 **`原生 vLLM 0.28.0` + MTP(5)**；本引擎本臂 MTP(1) 的对应数字是 **41.59**。
归因链：**环境切换 38.17→41.59（+8.9%）、MTP 深度 41.59→68.83（+65%）**——
⚠️ 但**前一步不是单变量**（env + `MI250_GATE_GEMV` + `--disable-custom-all-reduce` + splitKV 同时变），
**不得读成「原生 env 更快」的因果**。
⇒ 真正的缺陷是 `config/ports.conf:65` 把 nightly 的数字挂在了 0.28 的引擎描述上——
**已就地更正该注释**。**引用本臂单流数字必须同时给三个轴：引擎 × MTP 深度 × 负载标签。**

---

## 5. ⚠️ 对既有判词的两处更正（2026-09-21，留痕不静默改写）

> 仓库纪律：推翻既有结论时，在被推翻处留痕。以下两条曾被本技能写成硬结论，
> 经 `docs/GLM-5.3-Flash-Quark-INT8-施工单-2026-09-21.md` 与
> `docs/MI250X-AITER-INT4-内核复核-2026-09-17.md` 推翻。

### 更正 1 —— 「Quark int4 走 OCP MX ⇒ 要 fp4 硬件，CDNA2 没有」

- ~~原判词~~：把 CT W4A16 说成唯一可选，MXFP4 判为"机制不可达"。
- **实际**：vLLM 有**显式的 `Mxfp4MoeBackend.EMULATION` 回退**
  （`fused_moe/oracle/mxfp4.py:135`、`quark/quark_moe.py:1227-1229`），
  本机**实产过** MXFP4 产物（GPU 14.8 min）。
- ⇒ 正确表述是「**机制可达；本档的性能与质量未测**」。
  出处：施工单 §5.2④ L159-163（该文自己已列为撤回项，此前未回写本技能）。
- 附注：§3 表里「MXFP4 慢 5 倍」是**速度**判词，**不能**替这条**机制**判词。

### 更正 2 —— 「通用 CK 模板重编到 gfx90a 的路子对 int4 不存在」＋「W4A8-int8 是唯一可能赢的设计」

- ~~原判词~~：int4 自建内核判死；W4A8-int8 是唯一出路。
- **实际**（`docs/MI250X-AITER-INT4-内核复核-2026-09-17.md`）：
  - L21 原文「**可行**——不是能不能编，而是**三处工程 + 收益待测**」；
    L166-167「判死理由是『没有可用的 int4 内核可换』，**而不是『内核不值钱』**」。
  - 实测：CK 的 int4 设备代码（含 `b_scale` / per-group-**128** W4A16）**能为 gfx90a 编出
    `amdhsa-amd-amdhsa--gfx90a` 码对象**，跑 **58.3 → 127.9 TFlops**，数值经三证据验证正确。
  - 例程的 arch 限制是**主机侧 `if` 门**（`gemm_xdl_bf16_pk_i4_v3.cpp:201`），
    **不是设备码门**。
  - 缺的是三处工程：**S2 实例未 ship + S3 stage2 归约需改 + S4 vLLM 无接线**。
- **W4A8 被否证**：单流 **M=1 时 int8 ALU 利用率仅 0.3%**（M=2048 才 47–53%，L267）；
  另一候选 ck_tile flatmm 是 **prefill 取向**（L354-355）⇒
  **两条候选的受益场景都不是单流 decode**。
- ⇒ 结论应改为：**int4 自建内核未判死，卡在三处工程；但两条候选都不指向单流 decode 收益。**

### 更正 3 —— 「`port_qwen4exp_eagle_annotate.py` 让 prefix-cache 复用从 0 变 87%」

- ~~原判词~~（09-18）：补丁使命中 `0 → 3200/3686 = 87%`、多轮 TTFT `1.43–1.86 → 0.28–0.71 s`⇒ 于是**把默认打开并写进 launcher**（护栏⑤）。
- **实际**（09-21 **三 boot 配对**复测）：四项测量（同 prompt 顺序、P/Q 交替、多轮 TTFT、"命中"计数）**全部一致** ⇒ 该补丁对吞吐**中性**，只把启动日志里 **9 行警告**变成 0 行。
- 🔑 **上游默认本来就有跨请求复用**：同 prompt 连发，**第 3 次起整段跳过预填充**（2.51 s → 0.45 s）。那 9 行说的是"组被当成草稿组"，**不等于"复用为 0"**。
- **最可能的误因**：把**冷 boot 首个请求撞 Triton JIT** 当成了"没补丁就慢"——同一个坑在 09-18 另有 12.64 t/s 离群点记录（见 `references/60-…` §8 第 2 条）。
- 留痕：`docs/recipes/patches/qwen4exp-mtp-prefix-reuse.md` §3、`bench/ledger.jsonl` L48（L48 明确撤回 L44）。
- **可迁移的判语：警告消失 ≠ 行为改变。** 判"补丁生效"必须找一条**与警告无关**的独立尺子（这里是 prefill 耗时），并按配对 protocol 复测。

---

## 6. 死路（负结论也是资产）

**完整清单见 `references/50-dead-routes.md`（动手前必读）。** 不要花启动时间去重测
`data/*.json` 里 `status: dead` / `what_failed` 记过的东西。摘要：

- Ornith-1.5-397B **FP8** 在 vLLM/gfx90a 上 engine init 即死（**硅门**）⇒ 该模型本机
  只有 llama.cpp Q8_0 (8110) 一条路。
- AITER 的 MoE 路径与 MLA/CK decode 是 gfx942/950 专属；ROCm 10 上 AITER 仍翻不动。
  ⚠️ 但要把「AITER 不可用」**收窄到 MoE 通路**——`sparse_attn_indexer_kpool.py:1034/1062`
  的门**不是硬件门**。
- **QuickReduce C2+C3：开了就起不来**（2026-09-21 三臂实证，推翻"装了就可用"的默认假设）。A（stock 镜像）4.38/28.15/74.77 + 召回 6/6；A2（`-qr` 镜像、env 不设）4.18/26.64/73.13 + 6/6，两臂都选 `['PYNCCL']`，差 −2…−5% **落在跨 boot 漂移内 ⇒ 单次对拍不能说"中性"**；B（`-qr` + 三条 env）env 确实生效（`Custom quick allreduce: min size override = 0 MB`），但 `init_custom_qr` 在显存规划**之前**吃 **~9 GiB/卡** ⇒ `ValueError: Free memory on device cuda:5 (54.9/63.98 GiB) … less than desired (0.97, 62.06 GiB)`，8 worker 全拒启。要开必须 `util ≤ 0.858`，那时 KV 只剩 **~2 GiB/卡**（32k 档原本 8.17）⇒ **1M 上下文不可能**。全量证据 `hyperloom/reports/models/glm53-int4/qr-c2c3-verdict.md`。
- 27B BF16 MTP 三件套上的 prefix-cache 假设 —— 见 knobs。
- **engram 表进主机 RAM = 死路**（三段证据）；`DSV41_ENG_HOST_PREALLOC=1` 可行但不划算。
- **`case 256` head_dim 内核**：快 39× 但**算错 = NO-GO**；且**真锁是 `block_size` 不是 head_dim**
  （`rocm_attn.py:179-183` 只支持 16/32 + `_align_hybrid_block_size()` 把混合模型抬到 400/528）。
  「用 `hidden/heads` 推 head_dim 对 MLA/稀疏模型无效」。
- **TileLang mHC 在 gfx90a 上静默算错 `layer_input`**（`mhc.py::_has_tilelang_mhc()` 只排除
  gfx942、未排除 gfx90a；**非确定性**，单次对拍抓不到，必须连跑多次）。
  推论：**wave64 是 gfx9 的系统性风险面** —— 凡有 `tx<32` / `warpSize` /
  `__ballot(0xffffffff)` 假设的融合核，都要按「可能静默算错」验。
- 🔁 **同一个 mHC 闸已经咬过两次**（09-18 DSV4.1 = 补丁 ⑫；09-21 **8127 GLM-5.3-Flash Quark-INT8**
  被记成「数值已判坏、三个嫌疑」，实为**同一 bug 的第二棵树**）。**根因判据**：坏得**与量化无关**
  （两条不同 int8 线性核、eager/非 eager 全坏 + 贪心不可复现 + **权重侧审计全过**）⇒
  **先查每层都过的公共路径，别从「哪个量化坏了」出发**（这条顺序错了白烧 4 轮起服）。
- ✅ **接入门（零成本，必做）**：任何 config 带 `hc_mult` / `hc_sinkhorn_iters` / `mhc: true` 的模型，
  接入第一天就确认**它将要用的那一棵树**含 gfx90a 排除。MHC 使用者全集只有两族：
  `models/glm5next/*`、`models/deepseek_v4/{amd,xpu,cpu}`（`glm_moe_dsa`→`deepseek_v32` **不用**，
  所以 8121 与此无关，与它事实召回 6/6 自洽）。⚠️ **一台机器上有多棵 mhc.py**：容器挂载树 ≠
  宿主 editable 树（`src/vllm-master`）≠ 0.28 site-packages —— **只修一棵 = 其余照坏**，
  且宿主 editable 树**没有 launcher 预检兜着**，是纯盲区。机制化：
  `ROCm.AI/hyperloom/patches-local/apply_mhc_gfx90a.py --target <file> --check|--dry-run|--apply|--revert`。
  零改码对照法：分派点在函数体内读**模块级全局** ⇒ sitecustomize 设 `mhc.HAS_TILELANG_MHC=False`。
  证据链：`hyperloom/reports/models/glm53flash-int8/rootcause-mhc-tilelang.md`。
- ⚠️ **平台门 ≠ 能力门，但过门要用放行、不能用总闸**（09-21 · 8128 实测，零 GPU 判死一条杠杆）：
  `glm5_next`（GLM-5.3-**Flash**）与 `glm_moe_dsa`（GLM-5.3）**不是一个代码路径** —— 前者走新架构包
  `vllm/models/glm5next/{amd,nvidia,common}`，ROCm 分派到 `amd/`，其 `sparse_indexer.py` 里
  `if not rocm_aiter_ops.is_enabled(): raise "…only supported on AITER"` 是**这棵树里唯一一处 is_enabled 型门**
  （`forward_native` 只是 `return forward_hip(...)`，没有第二条实现可退）。放行 = 抄
  `model_executor/layers/sparse_attn_indexer.py` 已有的 `or on_gfx90a()` 写法（= 第 ⑧ 件补丁，单 hunk）。
  **绝不用 `VLLM_ROCM_USE_AITER=1` 过这道门**：`is_enabled()` 是总闸、子闸（`_MLA/_MHA/_LINEAR/_RMSNORM`）
  默认全 `True` 只靠 `master and 子闸` 关着，一翻就连带打开 gfx942/950 的 CK；而且实现体
  `v1/attention/ops/rocm_aiter_mla_sparse.py:841` 的 aiter 分支**排在 gfx90a 专项 Triton 分派之前**，
  那条 deepgemm fp8 内核在 gfx90a 上**编译不过**。判法只要两步 CPU：读那棵树的门 + 读分派顺序。
  ⚠️ 「只有一处门」**说过头了（撤回）**：同一分支里还有 assert 型平台门 ——
  `assert isinstance(…, DeepseekV32IndexerMetadata)`、`assert not use_fp4_cache,
  "Unfused FP4 Insert is not supported yet"`、`assert page_size % 16 == 0`。本臂都过了，
  但它们同样会按 config 触发 ⇒ 扫门要一起 grep `assert`，别只 grep `raise`。
- ⚠️ **放行一个平台门 = 接管它的整条未验收实现面**（09-21 学到的代价）：第 ⑧ 件治好「起不来」，
  但 `index_kpool=4` 走的分支**不经过**我们为 gfx90a 注册的那个 op（`grep kpool _aiter_ops.py` = 0 命中），
  而是走 `models/glm5next/amd/ops/kpool_compress.py`（**859 行 Triton，本机第一次被执行**）。
  ⇒ 这类臂**只能记 `experimental`**，不能说「已修好」。
- 🧮 **先用算术做减法，再去跑实验**（本轮最省钱的两个结论都来自纸面）：
  ① `index_topk=2048` ≫ 短 prompt（n≪2048）⇒ top-k 选择是平凡的（全选）⇒ **indexer 的 logits 再错
     也改变不了注意力输出** ⇒ 短 prompt 上出现坏位，嫌疑只能在写入/池化值或公共路径，不在选核；
  ② paged-KV 每次 block table 不同 ⇒ 1e-2 量级 logprob 抖动是**预期代价** ⇒「贪心两次不一致」**不能**
     单独作为坏臂判据（详见下一条）。
- 📐 **判据要先在健康对象上校准**（09-21 差点误判一次）：`probe_nll.py` 的「贪心两次必须逐字一致」
  是从**坏臂症状**反推的，本机从未有任何 vLLM TP8 臂通过它（含健康的 8121）⇒ 它是**阳性指标**、
  不是**必要条件**。量「单位置塌缩」用 `tools/probe_disaster.py`：`rate`/`phase`/`churn` 记**绝对位置**并扫
  mod 2/4/8/16/32/128；`invar` 用因果不变量（位置 i 的分布不该受其后续内容影响）分开
  **「模型算错」与「`prompt_logprobs` 取错行」** —— 这两种解释在 -19 这种症状下**长得一模一样**，
  不先分开就会整轮追错东西。
- ⚠️ **跨树对比 ≠ 单变量对照**：换镜像/换树时即使「只挂一个补丁」，两臂仍差着实现布局；
  把旁证写成单变量会让下一个人以为因果已钉死。真要单变量就在**同一底座**上开关那一个挂载。
  症状签名：权重装载成功、后端选择正常，**直到第一次 `_dummy_run` 才抛**；worker 全退后
  **容器可能仍显示 `Up`**（僵尸前端）⇒ 判活看端口/日志签名，不看 `docker ps`。
- 📏 **判据也要校准**（09-21 差点误判）：`tools/probe_nll.py` 的「贪心两次必须逐字一致」是**从坏臂的
  症状反推出来的**，本机**从未有任何 vLLM TP8 臂**（含健康的 8121）通过过它 ⇒ 它是坏臂的**阳性指标**，
  不是好臂的**必要条件**；当硬门用会把修对了的臂判成 FAIL。正确用法：**NLL 量级 = 硬门**，
  确定性 = 「同一 prompt 连打 k 次的灾难位率」另计。一般化：**用一条判据前，先查它在健康对象上
  是否成立过**（本仓 grep 留痕：只有坏臂的数据，就说明判据未校准）。
- 🧪 **`HSA_NO_SCRATCH_RECLAIM` / `HIP_FORCE_DEV_KERNARG` 已否证**（09-21）：曾被点名为「同输入不同输出」
  的头号待验修复项，实测 0918 镜像**早已烘焙两者**（`docker exec … env`）而症状照在。别再照这条开药。
- **不要在 isolated 微基准里排序**：结论会反（`CUSTOM allreduce` 输出静默变 `!!!`）。
- 🚩 **「CLI 接受性探针」只能用非法值触发 argparse**（09-21 实踩，白烧 3.5 分钟 LLM 额度）：
  `optimize --gpu-type <合法值> --model /tmp/不存在` 看着像"会报到模型路径就退出"，实际是
  **argparse 全过 → 会话目录真的建出来 → Claude 编排 agent 真的起来**（默认预算还按 2h 走）。
  报告里那条 V2 判据只在**没 source .env** 的裸环境下才安全。要证明 flag 合法，就传一个
  **不可能的值**去读 `invalid choice … (choose from …)`，并断言目标词在 choices 里。
- 🧯 **Hyperloom 的容器 ≠ 生产臂的容器**（09-21）：`hyperloom-local`（镜像 `-hl`）里 vLLM 的
  `glm5next/amd/sparse_indexer.py` 与 `model_executor/layers/mhc.py` 与 `-0918` 底座**逐字节相同**
  ⇒ 不含第 ⑧ 件与 mHC 回退。任何 `glm5_next` 权重进 Hyperloom 前必须先把这两件落进**那个镜像**
  （现已烘成 `rocm-ai/vllm:glm53-int4-hl-fl1`），否则 baseline 在第一次 `_dummy_run` 撞
  "Sparse attention indexer ROCm path is only supported on AITER."。同类坑：模型路径必须在容器
  挂载里 —— `-hl` 只挂了 `/mnt/kioxia-cm6-3t8/ai/models`，本权重在 `/mnt/stripe-3mix-3t2/models`，
  容器里 `ls` 不到；改挂载要 `docker commit` + 重建（可写层里有 Magpie 的 mi250x runner 注册，
  直接从底座 build 会丢）。

---

## 7. 数据文件与再生链路

- `data/scope.json` — **九轴词表 + 分层规范 + 8 条已知混淆点**（本技能新增的坐标系）。
- `data/arms.json` — 每条服务臂：身份、entry、定稿 args/env、实测指标、补丁、互斥。
- `data/environments.json` — 每个 `envs/*` 树的重建与自检。
- `data/knobs.json` — 跨臂开关的定稿值与决策规则（含 `applies_to` / `effect_by_scope` / `consumed_by`）。
- `data/patches.json` — 逐补丁的 apply/verify/revert 闸门。
- `data/ops.json` — 权重下载、起停观测、unsloth-studio。
- `data/recipe_kb.json` — Hyperloom RecipeKB 灌数索引。
- `data/vendor_references.json` — 厂商 recipe 折叠为**参考**：转移得过来的 flag 与
  厂商门控在 MI300X+ 的部分要分开。**参考 ≠ 本机验证**，绝不当 MI250X 结果引用。

### 7.1 这些文件怎么来的 —— 以及再生成时的陷阱

权威源是 `$AI/docs/recipes/**/*.md`（**51/55 入库**；未入库的只有 4 条最新的：`8121`、
`vllm-openai-rocm-nightly-0918`、`glm5next-quark-int8-launch-set`、
`glm53-flash-quark-int8-convert`——**也正是本轮查出从未进入 `data/*.json` 的那 4 条**，
所以那次抽取停摆既没有 JSON 兜底、也没有 git 兜底。旧文写「不在 git」不准确，已更正）。
`data/*.json` 是**由人阅读后合成**的：
`stack.notes`、`red_lines`、`status_note`、`path_or_target`、`rebuild`、`selfcheck` 都是散文与
浓缩命令，**不是正则能捞出来的文本**。实测：`rebuild`/`selfcheck` 与 markdown 代码块**逐行不等**。
⇒ **不要写"抽取器"**，那会把手工合成覆盖成垃圾并丢字段。管线是
`.tmp/kb_extract/*.json` → `scripts/build_skill_data.py`，抽取步是**阅读 pass，不是脚本**。

⚠️ **抽取曾停在 2026-09-15/16**，之后新增/改动的 5 条配方（`8121`、
`vllm-openai-rocm-nightly-0918`、`glm5next-quark-int8-launch-set`、
`glm53-flash-quark-int8-convert`、`vllm-fused-moe-tile-seeds`）是**手工补进 JSON** 的。
改了配方别以为技能数据会自动更新。

🔑 **同一文件被两个会话同时整段重写就会静默丢内容**（2026-09-21 实录）：`f8954bd`/`af1494e` 曾把"第三轮沉淀"写进本文件，随后并行会话的"第四轮"整段重写**把它删了**——判据是 `git show HEAD:SKILL.md | grep -c MI250_MOE_GEMV` **得 0**。⇒ 三条纪律：① **改完立刻 `git commit` + `grep` 复检**；② **长内容放报告，本文件只留结论 + 指针**（本报告就是按这条把细节放进 `hyperloom/reports/mi250x-skill-scope-audit-2026-09-21.md`）；③ 提交前先 `git status` 看**全量**输出——本轮就因 `| head -3` 截断而误判"无既存改动"，`git checkout` 打回了别人未提交的 09-21 更正（已还原）。

**`applies_to` 已在两个方向都落到位**（2026-09-21）：21 条 knob/patch 的权威 markdown 已回写
`applies_to`，`recipes/TEMPLATE.md` 也加了该段与**写法警告**——mini-YAML 解析器既不认嵌套 map
也不认独立 `#` 注释行（实测：嵌套 map 被折行嚼成垃圾串；注释行被吞进上一个键的最后一个值）。
**只有 `- 轴=值` 列表是安全的**。回写后 `--dir knobs|patches` 的 `scope_applicability` 漂移归零。

**改完配方 markdown 必须做的事**：把同一事实手工写回 JSON，然后跑闸口：

```bash
python3 /home/qiba/ROCm.AI/scripts/audit_skill_recipes.py --dir serving   # 也支持 environments|patches|knobs|ops
python3 /home/qiba/ROCm.AI/scripts/audit_skill_recipes.py --scope         # 适用范围自洽性
python3 /home/qiba/ROCm.AI/scripts/scope_match.py --orphans               # 双向差集
python3 /home/qiba/ROCm.AI/scripts/scope_match.py --env-check             # 臂↔环境 版本轴一致性
```

闸口查（只读）：markdown↔JSON 的 id 双向覆盖、可逐字映射的 frontmatter
（`status`/`verified`/`title_or_why`）、每个 `sources:` 在盘上存在（**证据根有四个**：
`$AI`、`$AI/docs`、`ROCm.AI`、`ROCm.AI/hyperloom`）、`consumers:`/`scope:` 能解析到**某个**配方 id
（不限种类——环境/补丁/开关/臂/操作都算消费方）、主题漂移。

**闸口本身被自证过**：主题漂移检查是用「往 `environments.json` 注入一条假的 `AITER_JIT_DIR`
声明」实测过的（检出后已回退）——所以它报绿是有意义的，不是摆设。

⚠️ **归一化是刻意有损的**：抽取会去掉中文词间空格与反引号（`vLLM 0.28.0` → `vLLM0.28.0`），
比较前要抹平，否则得到一堆假阳性——这个错误第一次跑就产生了 11 条误报。

---

## 8. Hyperloom RecipeKB 接线

本工作区另有一份由这些臂灌出的 RecipeKB：根 `/home/qiba/ROCm.AI/hyperloom/kb`
（7 级目录、`recipe.json` 行、hardware 固定 `mi250x`）。起优化会话前先指过去，
让 warm-start 读到本机历史：

```bash
export KNOWLEDGE_LOCAL_ROOT=/home/qiba/ROCm.AI/hyperloom/kb
```

配方变更后重生成（均幂等；路径相对 `/home/qiba/ROCm.AI/`）：
`scripts/seed_recipe_kb.py`（臂行）、`scripts/seed_reference_recipes.py`（厂商参考行 +
Ornith 臂折叠）、`scripts/note_emulation_boot.py`（boot 证据）、
`scripts/note_probe_crosscheck.py`（实测 TTFT/吞吐 + run 结局；并修复被优化器 CLOSE 抹掉的
`what_worked`/`what_failed`/`lessons`）、`scripts/note_agent_lane.py`（agent 线数字、
`TPOT ~= 45.4 + 3.97ms × ctx/1024` 律、`best_config` 重放键）；
再 `python3 scripts/verify_recipe_kb.py` 与 `python3 scripts/build_skill_data.py`。

---

## 9. 待办（别当成已解决）

**已闭环**：
- 第二轮——`applies_to` 回写 21 条权威 markdown + TEMPLATE；`env → 版本` join 用归一化解决
  （0/15 → **15/15**）并落成 `--env-check` 闸口（**首跑即抓出一处真错**：
  `llama.cpp_gfx90a-2026.9.8` 一棵树含多个 commit 子前缀，env 条目的 engine 轴曾写成单值）；
  Top-10 资产已收编进 `references/`。
- 第三轮——并发会话追加的两段「本会话新增（日期）」**重新分层**（删 78 行、逐条安置，
  硬 token 127/133 存活、6 项经核为省略号/空格造成的假阴性）；新建 `references/30-arms-detail.md`
  兑现 `scope.json` 里声明的 T3 归属；**§1.1/§1.2/§1.3 三处 T0 安全写法按权威配方
  `serve-start-stop-observe.md` §2 补上了限定**（组杀的 PGID 边界、worker 父进程链、
  cgroup `KillMode`、`card0` 是 BMC、预分配判据的正确语义）；新增**更正 3**（前缀复用撤回）。

仍开放：

1. ~~`8116` 单流数字不一致~~ ✅ 已裁决（见 §4）：非矛盾，是 `ports.conf` 把
   nightly+MTP(1) 的数字挂到了 0.28 的描述上；注释已就地更正。
1a. 🔴 **`8115` / `8116` / `8117` / `8119` 四条现役臂没有配方 markdown**（`$AI/tools/audit_recipes.py`
    报「serving 配方没有引用实体脚本」）⇒ 它们**进不了 `data/arms.json`**（硬加会造成孤儿条目、
    `--dir serving` 报红）。这是"臂表齐全"与"配方库齐全"之间的真实缺口，
    **补配方是前置动作**，不是加 JSON 字段能绕的。
2. **`serving` 配方仍无 `rocm:`/`torch:` 字段**（0/15）。版本轴目前由
   **臂 → env 配方** 这一跳供给并被 `--env-check` 锁住，不再靠人工抄写；
   彻底解决要在 serving frontmatter 加必填字段并改 `$AI/tools/audit_recipes.py` 的 `REQUIRED`。
3. ~~`aiter` a8w8 调优表 54 行未装进 env，是待回收的便宜杠杆~~
   → **该定性是错的，已更正并装表（2026-09-21）**。**代码级事实**（`aiter/ops/gemm_op_a8w8.py:656-663`）：a8w8 这条路径从 `a8w8_tuned_gemm.csv` **只消费 `splitK` 一个字段**，查不到就 `splitK=0`；而 54 行 gfx90a 的 splitK **全 = 0** ⇒ 装表**行为完全等价，性能恒等**。它的唯一作用是消掉每形一行 `not found tuned config …`（本机实测每次启动约 432 行），**这条日志本身是正确行为，不是待修的缺陷**——把它当性能杠杆是本技能的一处历史错误。
   ⚠️ 本轮同时发现**技能内部自相矛盾**：`data/knobs.json`（源自上游 `INT8-GFX90A.md`）
   一直写着「不要去补表，那是正确行为」，而 §9 与 `references/20` 却叫它"便宜杠杆"——
   我这轮照错的半边执行后才回查出来。**矛盾已在两处同时留痕。**
   已装进三个 env（各 580→634 行，独立复核通过），备份 + 一行回滚在
   `references/10-version-matrix.md` 的 a8w8 小节。
3a. ✅ 已装（2026-09-21 第三/五轮）：见 §7.1 与 `references/20` §5——**性能恒等**，只消噪音；回滚 = `cp envs/<e>/…/a8w8_tuned_gemm.csv.bak-20260921 …`（三个 env 各一份）。
4. **`moe_tune_w4a16.py` 基准夹具与生产形状不一致**（uint8 `[N,K/2]` vs 生产 int32 `[N,K/8]` + bf16 scale）
   ⇒ **其胜负数字在夹具修好前不能用于接线**。
5. **DSV4.1 转换记录仍有大块未沉淀**：已补 §4.1/4.8/4.10/4.15/4.16/4.19/4.21/4.22/4.24/4.28/4.30/4.33，
   但全文 2462 行 / §4.1–§4.39 只覆盖约一半。
5b. ~~其余 ~30 条已判级未安置~~ ✅ 第四轮已按「结论 + 指针」安置。
   **覆盖度量化**（41 个 harvest 小节 × 独有硬 token 命中率，≥60% 记为已收编）：
   **✅32 / 🟡8 / 🔴0**（唯一的 🔴 经核是 token 形式假阴性，实质内容已在 `references/60` §4）。
   8 条 🟡 是"结论已落、逐条行号明细仍在 `.tmp/harvest/docs-top.md`"——
   这是**有意的**：按 §7.1 的纪律，技能里只留结论 + 指针，不把 41 小节全文搬进来撑爆常驻上下文。
5b2. 🛑 **`--metric-audit` 现报 27 条吞吐数字缺负载标签**（advisory，不是 bug）。它是本轮把「38.17 vs 68.83 被当矛盾挂了三轮」这个教训机制化的产物。逐条补口径是后续活。
5c. **技能正文与 `data/*.json` 是两套覆盖**：24 个「已覆盖」判词里 **11 个只被 JSON 覆盖、
   正文 0 命中**（如 `llamacpp-tp-rccl-split-mode` 逐节引了 TP/RCCL 实测，正文完全没提）。
   判「是否已沉淀」必须同时看两处——本轮已在路由表里把 JSON 指过去，但正文索引仍不完整。
6. ~~`audit_log_paths.py` 漏检未修~~ → **该 todo 是过期的**：`/tmp` 写点检测早在 **2026-09-16 已修**（`TMP_WRITE_RE` 按写点判定 + `mktemp` 白名单），本轮实测全仓 `/tmp` 命中 = 0、8109 那 4 处也早已改掉。**但同一类缺陷在另一处仍在**：PID 消费方检查没豁免整行注释 ⇒ 19 条发现**全是假阳性**（launcher 头注释里的停服示例被当成消费方）。本轮已补豁免并登记 VOCAB （8115/8116/8117/8127/8119/8121），该闸口发现数 **19 → 4**，剩下 4 条是真的：8121 用容器名式无 `MODEL_KEY`；三条 8119 脚本用 `%H%M%S` 违反 `%H%M` 规范（**它们其实是同一个分钟级撞名隐患的反向解法，改动前需拍板，我没有擅自动 launcher**）。
7. ~~未提交/未 push~~ ✅ 已提交并推送：`cdfef9f..72f0a2a` → `origin/main`
   （49 笔，含并发会话的工作；推送前对**全部待推提交**做过敏感面扫描）。
   🛑 **推送需 `export PATH="$PWD/.tools/bin:$PATH"`**——`git-lfs` 装在 `.tools/bin`，
   不在 PATH 时 pre-push 钩子会让推送**静默失败**（`error: failed to push some refs`，
   看着像网络/权限问题，其实是 LFS）。本仓没有 sudo，这是唯一正解。
   ⚠️ 两个坑（仍适用于下次）：① `scripts/note_agent_lane.py` 是**会话前既有改动**，不属本次范围，别顺手带上；
   ② **本机有并发会话在同一工作区提交**（`def33b9` 曾把本技能尚未提交的回写一并带走）
   ⇒ 提交前先 `git status` **看全量、不要截断输出**（本轮就是 `| head -3` 截断导致误判，
   `git checkout` 打回了一处别人的未提交更正，已还原）。
8. ~~4 条新配方未入库~~ ✅ 已由并发会话 `def33b9` 提交；51→55 全部入库。
8bis. 🛑 **既存红线违规一处（非本次引入，已核实无泄密）**：
   `hyperloom/session/runtime/recipe_kb/.kb_preflight.json` **被跟踪**，
   而 §7 规定「`session/**/runtime/` 一律不入库」。它内容是 4 个布尔值、**无凭据**，
   且**早已在 origin/main**（引入于 `ed38799`），本次推送未新增暴露。
   ⇒ 待办：`git rm --cached` 并加 `.gitignore`；同时确认那条红线的真正理由仍然成立——
   **`runtime/kernel-agent.env.sh`（内含明文 API key）实测未被跟踪，红线核心守住**。
8ter. ⚠️ **`/home/qiba/ai` 有 29 笔提交未推 gitee**（其中 28 笔非本次工作）。
   已对待推集做过扫描（`session/runtime` 0 命中、凭据形态 0 命中、
   `config/unsloth-studio.env` 只是"凭据放哪"的说明无真凭据），
   **但 `gitee/taijizhang/rocm-server` 的可见性我无法核实，且那 28 笔不是我写的**
   ⇒ 未擅自推送。该仓制度「提交即推」要求补推，请人工确认后执行。
9. 🔑 **`data/*.json` 与配方 markdown 会因并发而漂移**：`--dir knobs|patches` 的"覆盖缺口"
   只报 INFO 不报红 ⇒ 新配方不会强制进 JSON。**本轮已把 markdown 侧 `applies_to` 的词表校验
   补上（`--dir *` 会发现越表值）**，但"新配方未合成进 JSON"仍无硬门。