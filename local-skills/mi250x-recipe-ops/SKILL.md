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
| `references/40-knobs-and-patches.md` | 自研 kernel、稀疏 split-K 开关、roofline 口径 | 调性能杠杆 |
| `references/50-dead-routes.md` | 死路与负结论（DSV4.1 线为骨干） | **动手前必读**，防止重做 |
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
- 功率：`560 W/module`，`/etc/amdgpu-powercap.conf`（`POWER_CAP_UW=560000000`）。
  **跨 run 比较前必须先对齐 cap**，报告数字必须带 cap 口径。
- 显存读数单位是 **bytes**：
  `paste <(seq 1 8) <(for c in /sys/class/drm/card{1..8}; do cat $c/device/mem_info_vram_used; done)` — 空闲 ≈ 1.0e7

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
- 🛑 **不碰 `/opt/rocm` 的 alternatives**。宿主 `/opt/rocm-7.2.4` 是**软链指向本盘解包树**
  （非 apt 完整安装）——**apt 会透过软链覆盖解包树，直接打挂正在跑的 `vllm_0.28.0_rocm72`**。
- 🛑 **`HIP_VISIBLE_DEVICES` 会泄进 ROCR 掩码子环境**，导致 `import vllm` 直接抛。
- **public 仓库**：不写 `git add hyperloom/session`；`session/**/runtime/` 一律不入库
  （`runtime/kernel-agent.env.sh` 是 0644 内含 API key 明文）。
- **只在需要时起服**；测量/验证一跑完立即停服，不留常驻服务占着 8 张卡。

### 1.3 Step 0：起任何东西之前

1. 哪些 die 被占（§1.1 的显存命令）。
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

⚠️ **本节曾只到 8121 且漏 8115/8116/8117/8119** —— 权威是 `config/ports.conf` + launcher 实物。

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
| **8115** | Ornith-1.5-397B **Quark INT8 W8A8**（出厂/保守臂） | vllm 0.28 | active |
| **8116** | Ornith-1.5-397B **CT-Int4 W4A16** + MTP | vllm 0.28 | active |
| **8117** | 同 8115 checkpoint + 三个杠杆（AITER int8 补丁 / SPEC=5 / `--no-enable-prefix-caching`） | vllm 0.28 | active |
| **8119** | DeepSeek-V4.1 int4（**与 8121 共用同一棵补丁树**） | vllm nightly-0918 | active |
| 8121 | GLM-5.3-CT-Int4-W4A16 402 GB（**本机自转**）TP8 | vllm nightly-0918 | active |
| **8127** | **实验臂专用端口**（09-18 两会话撞 8117、PID 文件互相覆盖后定的规矩） | — | — |
| 8203 | Qwen2.5-1.5B smoke | vllm 0.28 | active |
| 8301/8302 | 27B Fable splitKV / dflash2 | vllm 0.28 | experimental |
| ~~8207~~ | ROCm 10.0.0 A/B 臂的**墓碑，勿复用** | — | 已退役 |

**选型**：单流最快 → 8107 llama.cpp Q4_K_XL；OpenAI API + 大 KV 池 → 8107 vLLM BF16；
唯一常驻大模型 → 8112。完整理由见 `$AI/docs/recipes/README.md` §5。

⚠️ **`8116` 的单流数字两处不一致**：`config/ports.conf` 记 **38.17 t/s**，
SKILL 第四轮记 **68.83 t/s**（MTP(5)@256K）。**未对齐口径前不得引用任一个**——
这正是"数字必须带口径"的反面教材，已列入 §9 待办。

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

---

## 6. 死路（负结论也是资产）

**完整清单见 `references/50-dead-routes.md`（动手前必读）。** 不要花启动时间去重测
`data/*.json` 里 `status: dead` / `what_failed` 记过的东西。摘要：

- Ornith-1.5-397B **FP8** 在 vLLM/gfx90a 上 engine init 即死（**硅门**）⇒ 该模型本机
  只有 llama.cpp Q8_0 (8110) 一条路。
- AITER 的 MoE 路径与 MLA/CK decode 是 gfx942/950 专属；ROCm 10 上 AITER 仍翻不动。
  ⚠️ 但要把「AITER 不可用」**收窄到 MoE 通路**——`sparse_attn_indexer_kpool.py:1034/1062`
  的门**不是硬件门**。
- vLLM TP8 QuickReduce 默认值、27B BF16 MTP 三件套上的 prefix-cache 假设 —— 各见 knobs。
- **engram 表进主机 RAM = 死路**（三段证据）；`DSV41_ENG_HOST_PREALLOC=1` 可行但不划算。
- **`case 256` head_dim 内核**：快 39× 但**算错 = NO-GO**；且**真锁是 `block_size` 不是 head_dim**
  （`rocm_attn.py:179-183` 只支持 16/32 + `_align_hybrid_block_size()` 把混合模型抬到 400/528）。
  「用 `hidden/heads` 推 head_dim 对 MLA/稀疏模型无效」。
- **TileLang mHC 在 gfx90a 上静默算错 `layer_input`**（`mhc.py::_has_tilelang_mhc()` 只排除
  gfx942、未排除 gfx90a；**非确定性**，单次对拍抓不到，必须连跑多次）。
  推论：**wave64 是 gfx9 的系统性风险面** —— 凡有 `tx<32` / `warpSize` /
  `__ballot(0xffffffff)` 假设的融合核，都要按「可能静默算错」验。
- **不要在 isolated 微基准里排序**：结论会反（`CUSTOM allreduce` 输出静默变 `!!!`）。

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

**改完配方 markdown 必须做的事**：把同一事实手工写回 JSON，然后跑闸口：

```bash
python3 /home/qiba/ROCm.AI/scripts/audit_skill_recipes.py --dir serving   # 也支持 environments|patches|knobs|ops
python3 /home/qiba/ROCm.AI/scripts/audit_skill_recipes.py --scope         # 适用范围自洽性
python3 /home/qiba/ROCm.AI/scripts/scope_match.py --orphans               # 双向差集
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

## 本会话新增（2026-09-21：前缀复用口径更正 · 三 boot 配对 · 护栏全谱）

### 前缀复用：上游默认本来就有，标注补丁只消警告（**正面撤回**）
- 本臂（8107 Qwen3.8-Flash-Next BF16）启动时会打 9 行
  `… will be treated as a draft group … prefix-cache reuse across requests will be disabled`。
  **那不等于复用为 0**：上游默认下同 prompt 连发，**第 3 次起整段跳过预填充**（2.51 s → 0.45 s）。
- `patches/gfx90a/port_qwen4exp_eagle_annotate.py` + `VLLM_QWEN4EXP_EAGLE_ANNOTATE=1` 让警告 9→0，
  但**三 boot 配对复测四项测量全部一致**（同 prompt 顺序、P/Q 交替、多轮 TTFT、"命中"计数）
  ⇒ 它对吞吐**中性**，只是消掉误导性警告。
- **撤回**：09-18 写的"命中 0→87%""多轮 TTFT 1.43–1.86→0.28–0.71 s"不可复现，
  最可能是把**冷 boot 的 Triton JIT 尖峰**当成了"没补丁就慢"。留痕见
  `docs/recipes/patches/qwen4exp-mtp-prefix-reuse.md` §3 与 `bench/ledger.jsonl` L48。

### 判据只认耗时（这条最容易记错账）
- `vllm:prefix_cache_hits_total` 与引擎周期行 `Prefix cache hit rate: X%` 在**本版不区分内容**：
  全新 prompt 也报 ~90%（实测 Δh 恒 = 块数×400）。拿它们当跨请求复用判据会得出任意结论。
- 可信判据 = **同 boot 内新 prompt vs 重复 prompt 的预填充耗时**（本机 6.7k tok：2.51 / 0.45 / 1.59 s）。
- 探针 prompt 的唯一标签要**每次运行唯一且进每一行**；只放开头会命中共享正文（假阳性）。
- 真冷基准要把 salt 放**最前**（放末尾会让"冷"那一次其实命中旧缓存）。

### 报数协议增补（原三硬门之外）
**必须同形状预热**（冷 boot 前若干请求会撞 `Triton kernel JIT compilation during inference`：
`_qsa_sparse_paged_gqa_splitk_kernel` / `_rejection_kernel` / `_resample_kernel`；实测 12.64 t/s ≈30 s/请求，
spread 1.1%→87%）；**必须带 acceptance + step ms**（本臂两条 workload step 同为 ~39–40 ms，
29% 的 TPS 差全来自接受率 89.9% vs 61.2%）；**必须核他会话 CPU/内存**（实测并行会话
`convert_dsv41_ct_int4.py`：1305% CPU、RSS 56.7 GB、主机 free 1 G + swap 10 G，显存却看着"没人占"）；
正确性容差要建在引擎先天 `|Δlogprob| ≤ 0.11` 之上。

### 护栏全谱与"门必须故意触发一次"
launcher 内 7 道 fail-closed 门：①模型目录 ②`nc` 端口门 ③HF offline ④功率档只读校验（560 W）
⑤草稿组标注准入（`--require`）⑥8 die 显存前置门 ⑦同端口实例存活门（核 `/proc/<pid>/cmdline`，
防 PID 回收误拒）。教训：本臂 QR 那条准入断言曾因写了**不存在的补丁路径**而退化成
"QR 永远开不起来"且长期无人发现（默认 QR=0 掩盖了它）⇒ 每道门都要**故意触发一次**。

### 起停 / 验证 / 实体分叉
- 停服必须三步（本底座 TERM 会挂住，worker 报 `FileNotFoundError: /psm_*`）：见
  `docs/recipes/ops/serve-start-stop-observe.md`；**别用 `kill -KILL -$pid` 负号进程组**（已两次打死自己的 shell）。
- **不占 GPU 验证 launcher 接线**：`VLLM_PYTHON=<桩脚本>` + 私有端口 + `VRAM_FREE_MIN_GIB=0` 干跑，
  核对 env 与落盘路径（本会话靠它证明 6 项接线生效，全程没碰 GPU）。
- 同一条臂有 **3 份实体**（基脚本 / `launcher/favor/…_final_….sh` 薄封装 / `/mnt` 镜像）⇒
  改动只落基脚本 + `md5sum` 对账 + 侧车软链，否则会重演覆盖事故。
- 8107 日志已改落 `logs/flash-next/server-<port>-<ts>.log`（+ `.current` 软链），
  取日志请读 `.logpath` 侧车，**别用旧的 glob**（会静默读到旧日志）。

### 数字口径（8107 BF16，同一臂三个时点）
09-05 单流 88.6 / Phase2 296.86（560 W、SPEC=3、batched 8192）；覆盖期 92.07、93.16；
合并 8 条臂级 env 后 94.85 / 94.62 / 95.29（acceptance 89.9%、step 38.9–39.2 ms）；
KV 池 719,056 tok（2.74×）。**那 8 条 env 性能中性**；`VLLM_ENABLE_V1_MULTIPROCESSING=0`
在 vLLM master 这版**并没有**把 EngineCore 并回 APIServer（仍有独立 `(EngineCore pid=…)` 行）。

## 9. 待办（别当成已解决）

**本轮（2026-09-21 第二轮）已闭环**：`applies_to` 回写 21 条权威 markdown + TEMPLATE；
`env → 版本` join 用归一化解决（0/15 → **15/15**）并落成 `--env-check` 闸口
（**首跑即抓出一处真错**：`llama.cpp_gfx90a-2026.9.8` 一棵树含多个 commit 子前缀，
env 条目的 engine 轴曾写成单值）；Top-10 资产已收编进 `references/`。

仍开放：

1. 🔴 **`8116` 单流数字两处不一致**（`ports.conf` 38.17 vs 技能第四轮 68.83）——
   **对齐口径前不得引用任一个**。
2. **`serving` 配方仍无 `rocm:`/`torch:` 字段**（0/15）。版本轴目前由
   **臂 → env 配方** 这一跳供给并被 `--env-check` 锁住，不再靠人工抄写；
   彻底解决要在 serving frontmatter 加必填字段并改 `$AI/tools/audit_recipes.py` 的 `REQUIRED`。
3. **`aiter` a8w8 调优表里 gfx90a 的 54 行仍未装进 env**（三个 env 的 `a8w8_tuned_gemm.csv`
   只有 gfx942=26 + gfx950=553）⇒ 生产日志常年 `not found tuned config … will use default config`。
   纯文件操作（CPU），**任何 aiter INT8 计时/对比之前应先补**。
4. **`moe_tune_w4a16.py` 基准夹具与生产形状不一致**（uint8 `[N,K/2]` vs 生产 int32 `[N,K/8]` + bf16 scale）
   ⇒ **其胜负数字在夹具修好前不能用于接线**。
5. **DSV4.1 转换记录仍有大块未沉淀**：本轮补了 §4.1/4.8/4.10/4.15/4.16/4.19/4.21/4.22/4.24/4.28/4.30/4.33
   的要点，但全文 2462 行 / §4.1–§4.39 只覆盖约一半。
6. **`tools/audit_log_paths.py:86` 的漏检未修**（正则只匹配带引号赋值 ⇒ `>/tmp/` 与无引号赋值漏检却报绿）。
7. **本轮所有技能/脚本改动 + 21 条配方回写均未提交**；
   ⚠️ 注意 `scripts/note_agent_lane.py` 是**会话前既有改动**，不属本次范围，别顺手带上。
8. **4 条新配方 markdown 未入库**（`8121`、`vllm-openai-rocm-nightly-0918`、
   `glm5next-quark-int8-launch-set`、`glm53-flash-quark-int8-convert`）——
   按制度「`??` 不过夜」应尽快 commit；它们也正是本轮查出从未进 `data/*.json` 的那 4 条。
## 本会话沉淀（第三轮，2026-09-21 **重做**：上一次被并行会话整段重写吃掉）

> ⚠️ 事故留痕：`f8954bd`/`af1494e` 两笔曾把本节写进技能，随后并行会话的「第四轮」重写把内容**静默删除**
> （HEAD 里 `MI250_MOE_GEMV` 命中 0）。**同一文件被两个会话同时整段重写就会丢内容** ⇒ 改完立刻
> `git commit` + `grep` 复检；长内容优先放报告，技能里只留结论 + 指针。

### MoE 专家 GEMV（目前唯一已收回的 kernel 级杠杆）
- 模块 `quark-int8/moe_gemv_patch/mi250_moe_gemv_gs.py`；开关 `MI250_MOE_GEMV=1`、
  `MI250_MOE_GEMV_MODULE=mi250_moe_gemv_gs`、`MI250_MOE_GEMV_KERNEL`（v3 = scale 提出 k 循环）、
  `_BOTH`、`_DEBUG`；`PYTHONPATH=/patches/moe_gemv` 由 launcher 注入。
- 实测：gemm1 **8.7×** / gemm2 **5.0×**；单流 6.38–6.81 → **9.60–10.71 tok/s**；conc32 聚合 33.8 → **59.1**；召回 6/6。
- `mi250_moe_gemv_v2.py` 是**被证伪**的那版（≈等于不开），别当可用模块；三处副本哈希由
  `verify_patches.py ②` 守（`gs=af079db138ab` / `v2=c96264af84c1` / `v3=07b0d78f40b0`）。
- ✂️ **已收回**：decode 归属表里 MoE GEMV 只占 **1.8%** 步时间，别在这里找收益（`moe-gemv-scale-hoist.md`）。

### 稀疏注意力 split-K（`MI250_SPARSE_SPLITK`）——与 0.28 的 split-KV 不是一回事
- 机制：一条 launch 把 `(query, split)` 当行（`_splitk_make_indptr` 造 `[M*S+1]` indptr、行序 `r=i*S+s`），
  再 `_splitk_merge` 做 LSE 合并；补丁 `quark-int8/dcp_patches/0009_gfx90a_sparse_splitk.patch`。
- 开关 `MI250_SPARSE_SPLITK=8`（**默认 0**）/ `MI250_SPARSE_SPLITK_MAXM=8`。
- 内核 7.2×（M=1）/ 2.4×（M=4）、与 S=1 逐位一致（bf16 1 ulp）；但端到端单流 **−12%**
  （NCCL 141→255 µs、elementwise/copy 调用 1332→3439/step）⇒ 默认必须保持 0。
- ⚠️ 静默错：低层 `_rocm_sparse_attn_prefill_ragged_triton` 是**返回** out（内部 `empty_like`），
  读预分配缓冲会得到 out 全 0 而 lse 正常 —— 看起来没崩，结果全错。

### Roofline 口径：本机的慢**不是带宽**（别再按带宽解释）
- `T_mem(mi250x)` @8 GCD / isl=osl=1024 / conc=32 = **859.0 tok/s**；同参数 mi300x 2779.3
  ⇒ **比值 0.309 是防「按 MI300X 口径假通过」的判据**。
- 实测（同一把尺子）：单流 @ctx≈800 **1.19%**、conc8 4.82%、conc32 **6.93%**；权重流量只有
  **18.6 GB/s/rank = 峰值 1.1%** ⇒ 受限在层内串行/启动延迟。正确表述：**层内串行开销吃掉 93% 的访存预算**。

### RecipeKB 回填与三个静默坑（GLM-5.3 int4 / mi250x）
- 入口 `python3 scripts/note_glm53_int4_kb.py`（`--dry-run` 可预演）；必须走
  `LocalRecipeStore.put_recipe`，手写 `recipe.json` 会让 `history/vN` 与 `version` 脱节。
- ① `remaining_gaps` 条目必须是 **dict**（`description`/`metrics`），写字符串被**静默丢弃**；
  ② `kernel_optimizations` 是**定长 dataclass**，键名不对得到**一串全零**条目；
  ③ `best_config.extra_envs` 会被 warm-replay **当环境变量注入** ⇒ 别写说明文字。
- `root:root 0600` 的槽位宿主读不到 ⇒ `local_store.search()` 抛 `LocalRecipeStoreError`；
  容器内 `chown -R 1000:1000` + `chmod 644` 修。
- warm-replay 置信门 `_DEFAULT_WARM_REPLAY_MIN_CONFIDENCE = 0.7`；recipe 的 `what_failed` 会被注入
  explore 的 rejected 账本 ⇒ **负结论写进去等于省一次重测**。

### QR C2+C3：**开了就起不来**（三臂实证 2026-09-21；推翻了「装了就可用」的默认假设）
- A（stock 镜像，env 不设）4.38 / 28.15 / 74.77 + 召回 6/6；A2（`-qr` 镜像，env 不设）4.18 / 26.64 / 73.13 + 6/6，
  两臂都选 `['PYNCCL']`；A2 对 A 差 −2…−5%，落在 cross-boot 漂移内 ⇒ **单次对拍不能说「中性」**。
- B（`-qr` 镜像 + 三条 env）：env 确实生效（日志 `Custom quick allreduce: min size override = 0 MB`），
  但 `init_custom_qr` 在显存规划**之前**吃 ~9 GiB/卡 ⇒
  `ValueError: Free memory on device cuda:5 (54.9/63.98 GiB) ... less than desired ... (0.97, 62.06 GiB)`，8 worker 全拒启。
  要开必须 `util ≤ 0.858`，那时 KV 只剩 **~2 GiB/卡**（32k 档原本 8.17）⇒ **1M 上下文不可能**。
- 复跑：`SKIP_A=1 SKIP_A2=1 bash quark-int8/qr_ab_watch.sh`；全量证据 `reports/models/glm53-int4/qr-c2c3-verdict.md`。
