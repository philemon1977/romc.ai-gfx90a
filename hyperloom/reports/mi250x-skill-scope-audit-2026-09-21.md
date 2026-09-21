# mi250x-recipe-ops 资产收割与适用范围分层 — 报告

日期 2026-09-21 ｜ 执行：DSH 会话 ｜ 范围：`/home/qiba/ai` 全工作区审计 + 技能重构

---

## 0. 一句话

审计了 `/home/qiba/ai` 的 **154 个 md + 53 条配方 + 82 个 tools 脚本 + 37 项 config**，
把技能从「按会话日期堆叠的单文件」改成「**九轴适用范围分层** + 常驻索引 + 按需 references」，
补齐了 5 条从未进入机器可读层的配方，修掉 3 处闸口缺陷（含 2 处从未生效的死检查），
并按仓库纪律**留痕更正了两处既有判词**。

---

## 1. 审计发现的硬缺陷（都可验证）

| # | 缺陷 | 证据 | 处置 |
|---|---|---|---|
| 1 | **5 条配方从未进入 `data/*.json`** | `8121`(serving)、`vllm-openai-rocm-nightly-0918`(env)、`glm5next-quark-int8-launch-set`(knob)、`glm53-flash-quark-int8-convert`(op)；`.tmp/kb_extract` 停在 09-15/16 | 手工合成后补入（阅读 pass，非脚本） |
| 2 | **闸口对 serving 零覆盖** | `audit_skill_recipes.py --dir` 只接受 `environments\|patches\|knobs\|ops` | 加 `serving`（`arms.json` / 主键 `recipe_id`） |
| 3 | **闸口 `consumers` 检查是死代码** | `arms = {a.get("id") …}`，而 arms.json 的键是 `recipe_id` ⇒ 集合恒为 `{None}`；且只在 `--verbose` 打 info | 修真 + 改成真 finding；并把语义修正为「解析到**任一**配方 id」（原来只认臂，把 env→env、patch→knob 全误报） |
| 4 | **serving 审计路径直接崩** | `frontmatter()` 遇 `key:` 后跟缩进续行时 `fm[key]` 是空列表 ⇒ `[-1]` IndexError | 空列表退化处理 |
| 5 | **证据根只有一个** | 8121 合法引用 `ROCm.AI/hyperloom/reports/...`，被报 3 条"来源不存在" | 证据根扩到 4 个；另修 `标签：绝对路径` 与绝对路径解析 |
| 6 | **技能臂表漏 5 条在册臂** | `config/ports.conf` 登记到 8127；技能表曾只到 8121 且漏 8115/8116/8117/8119 | 表已补全，并标注"权威是 ports.conf + launcher 实物" |

---

## 2. 分层设计（本轮核心交付）

### 2.1 九轴受控词表 → `data/scope.json`

`host` / `driver` / `rocm` / `torch` / `engine` / `arch` / `model` / `quant` / `topology`，
共 50 个词表项，每项带 `label` / `facts` / `evidence`。

分层：**T0** 通用（硅·驱动·纪律）→ **T1** 版本（engine/rocm/torch）→
**T2** 模型×量化 → **T3** 组合（臂与单次实测）。

### 2.2 三字段分家（本轮最关键的设计决定）

| 字段 | 语义 | 用在 |
|---|---|---|
| `applies_to` | 九轴**适用判定**谓词 | **匹配** |
| `consumed_by` | 真正的消费臂 id（旧 `scope`，含否定消费者与散文） | 溯源 |
| `effect_by_scope` | 某 scope 下是正是负 | 判收益 |

**为什么必须分家**（实测数据，不是设计偏好）：

- `consumed_by` 反查适用性会给出**错误的空集**：`8121` 被 **0 个 knob** 认领，
  而它实际适用 6 个。65 条 knob→臂边里 **63 条（97%）是单向的**——边只存在于臂的
  `related:` 里，从未回填到 knob 的 `scope:`。改用谓词后这 6 条全部找回。
- 该字段还被当成阅读日志：混入 knob id、散文（`"离线加载（唯一正解入口 …）"`）
  与**否定**消费者（`"8111-…（不接投机，仅为记录）"`）。
- knob 的谓词一度写得太窄（钉死在单个引擎版本上），会让"换版本还能不能用"得到
  **错误的否定答案** ⇒ 引入 glob（`vllm-*`）与 `preset`。

### 2.3 可运行的匹配器 → `scripts/scope_match.py`

分层若只写成文档，agent 照样错配。所以落成机制：

```bash
python3 scripts/scope_match.py --arm 8121-…        # 以现成臂为目标
python3 scripts/scope_match.py --engine … --quant … # 直接给轴
python3 scripts/scope_match.py --orphans            # 双向差集闸口
python3 scripts/scope_match.py --axes               # 词表
```

轴的三态：**具体值** = 有限定；**`"*"`** = 明确任意；**缺失** = 未知（不否决，但标记"判定不完整"）。

`--orphans` 是双向差集，两种方向意义不同：
`适用但无人记录`（可能没试过）vs **`记录了却判不适用`（危险矛盾——历史与规则打架）**。

### 2.4 SKILL.md 瘦身 + references 分层

| | 行数 | 字节 |
|---|---|---|
| 原 SKILL.md | 726 | 58,079 |
| 新 SKILL.md | 461 | ~31,000 |
| `references/`（5 个文件） | 651 | 53,565 |

拆分**零丢失**，经两道验证：
- **A** 机械搬运部分 495/495 行逐字保留；
- **B** 33 个关键标识符/路径/命令全部存活。

（验证器第一版只会问"这行还在吗"，把**有意改写**也报成丢失——改写与丢失必须分开判，
见 `.tmp/harvest/verify_split.py` 的注释。）

---

## 3. 新增的 T0/T1 事实（技能里此前 0 命中）

| 事实 | 出处 |
|---|---|
| **amdgpu-dkms 6.16.13 装在内核 6.8.0-139 上**；`uname -r` 与 `/sys/module/amdgpu/version` 不等是正常的 | sysfs + `modinfo` 实测 |
| **GPU NUMA 接线 4+4**：card1-4→node0，card5-8→node1（2×EPYC 7413） | sysfs `numa_node` 实测 |
| 🛑 **TP8 全 die 同步计算曾三次整机断电** ⇒ 功率 cap 只读不改 | `config/gpu_mi250x_tp8.conf:49` |
| **`/opt/rocm-7.2.4` 是软链指向本盘解包树**；apt 会透过软链覆盖它、打挂正在跑的 env | `build_vllm_master_gfx90a.sh:31-40` |
| **`config/env-lock/*.txt` 不含 ROCm 版本**（8 个文件 0 命中）⇒ 版本三元组要三方拼 | env-lock 实测 |
| **测量有效性两前提**：`workspace-write` 沙箱下 `/dev/shm` 不可写（曾致 SE-Bench 假 FAIL 12/17）；指令预算 65536 会**整份丢掉 AGENTS.md** | CLAUDE.md §0.1b；`tools/instruction_budget_check.sh` |
| **`rocprofv3` 计时地板 ≈ 4.96 µs**，会污染一切 profile 归因 | `MI250X-GLM-5.3-Flash-本机运行全记录` |
| **闸口全绿 ≠ 没有红线级问题**；`audit_log_paths.py:86` 正则只匹配带引号赋值 ⇒ 漏检却报绿 | `脚本评审-2026-09-16.md` |

---

## 4. 留痕更正的两处既有判词（§5 of SKILL.md）

| # | 原判词 | 实际 | 出处 |
|---|---|---|---|
| 1 | 「Quark int4 走 OCP MX ⇒ 要 fp4 硬件，CDNA2 没有」（判 MXFP4 机制不可达） | vLLM 有显式 **`Mxfp4MoeBackend.EMULATION`** 回退，本机**实产过** MXFP4 产物（14.8 min）⇒ 正确表述是「**机制可达、性能与质量未测**」 | `GLM-5.3-Flash-Quark-INT8-施工单-2026-09-21.md` §5.2④ |
| 2 | 「通用 CK 模板重编到 gfx90a 的路子对 int4 不存在」＋「W4A8-int8 是唯一可能赢的设计」 | 源文明写「**不判死、可行**」；CK int4 设备代码**实测能为 gfx90a 编出码对象**并跑 58.3→127.9 TFlops；arch 限制是**主机侧 `if` 门不是设备码门**。且 W4A8 被否证（**单流 M=1 时 int8 ALU 利用率仅 0.3%**） | `MI250X-AITER-INT4-内核复核-2026-09-17.md` L21/L166-167/L267 |

---

## 5. 资产裁决（子代理普查，`/home/qiba/ai`）

| 组 | 项数 | harvest | already-covered | skip |
|---|---|---|---|---|
| `docs/` 顶层 md | 76 | **41** | 20 | 15 |
| `docs/research` + `research-notes` | 21 | 18 | 1 | 2 |
| `config/` | 37 | 11 | 0 | 13 |
| `tools/` | 82 | 27 | 1 | ~54 |
| `CLAUDE.md` + `AGENTS.md` | 2 | 2 | 0 | 0 |

**元发现**：`data/*.json` 与 SKILL.md 是**两套覆盖**——24 个「已覆盖」判词里
**11 个其实只被 `data/*.json` 覆盖、SKILL 正文 0 命中**。判「是否已沉淀」必须同时看两套。

**优先级 Top-10**（可复用价值 × 缺口）：
1. `GLM-5.3-Flash-Quark-INT8-施工单` + `MI250X-AITER-INT4-内核复核`（唯二要求更正既有判词）
2. `MI250X-GLM-5.3-Flash-本机运行全记录`（R1–R14、9 行悬崖、4.96 µs 地板、ngram `n_min` 门槛）
3. `MI250X-attention-hd256-内核缺口`（`case 256` NO-GO 防重做；**真锁是 `block_size`**）
4. `DeepSeek-V4.1-Flash-CT-INT4-W4A16-转换记录`（§4.1–4.33 约 1900 行未沉淀）
5. ROCm 10 三件（最大版本维度）
6. `MI250X-indexer-非AITER-Triton路径`
7. `MI250X-hd256-customPA-静态审计`
8. `MI250X-vLLM-优化方案` + `safetensors适配清单`
9. `MI250X-会话经验总结`
10. `脚本评审` + `脚本归档索引` + `全仓评审与规划`

---

## 6. 本轮**未**完成（诚实清单）

1. **Top-10 资产的具体收编**：只把**更正与最高价值摘要**写进了 SKILL.md / references；
   41 个 harvest 小节的逐条要点仍在 `.tmp/harvest/docs-top.md`，未逐条落库。
2. **`applies_to` 未回写进 `docs/recipes` 的权威 markdown**：本轮把它变成**可见的漂移**
   （闸口 TOPICS 新增 `scope_applicability`，10 条 knob 全部报出），并记进
   `data/scope.json > gaps`；回写动作未做。
3. **`rocm`/`torch` 轴的可达性**：serving 配方 **0/15** 有这两个字段，且 `env:` 解析不到
   environment 配方 id（实测 0/15）⇒ arm→env→版本 的三跳 join 第一跳就断。
   已在 `scope.json > gaps` 记为 `env-axis-unresolvable` / `no-rocm-torch-in-serving`。
4. **`8116` 单流数字的冲突**（38.17 vs 68.83）未裁决。
5. **改动未提交**。

---

## 7. 本轮新增/修改的文件

| 路径 | 动作 |
|---|---|
| `local-skills/mi250x-recipe-ops/SKILL.md` | 重构（726→461 行）：§0 分层规范 + T0 主机层 + T1/T2/T3 速查 + §5 两处更正 + §9 待办 |
| `local-skills/mi250x-recipe-ops/references/*.md` | 新增 5 个（00-T0-host / 10-version-matrix / 20-model-quant-matrix / 40-knobs-and-patches / 50-dead-routes），逐字搬运 + scope 头 |
| `local-skills/mi250x-recipe-ops/data/scope.json` | 新增：九轴词表、4 层、preset、8 条混淆点、frontmatter 映射、3 条 gaps |
| `local-skills/mi250x-recipe-ops/data/{arms,environments,knobs,ops,patches}.json` | 补 4 条缺失条目；53 条全部加 `applies_to`；加 `consumed_by` / `effect_by_scope` |
| `scripts/scope_match.py` | 新增：可运行的九轴匹配器 + `--orphans` 双向差集闸口 |
| `scripts/audit_skill_recipes.py` | 修 4 处缺陷；新增 `--scope` 校验；新增 `--dir serving` |
| `.tmp/harvest/*` | 审计与验证中间物（`docs-top.md` / `research-config-tools.md` / split & verify 脚本） |

---

# 第二轮（同日 13:40–14:05）：闭环 + Top-10 收编

## 1. `applies_to` 回写进权威 markdown（21/21）

10 条 knob + 11 条 patch 的配方 frontmatter 全部加上 `applies_to`；`recipes/TEMPLATE.md`
同步加该段。**回写后 `--dir knobs|patches` 的 `scope_applicability` 漂移归零**——
第一轮"把它变成可见缺口"的动作在这里闭合。

### 过程中踩的两个坑（都写进了模板与脚本 docstring）

1. **两个 mini-YAML 解析器都不支持独立 `#` 注释行**：既不是 `^- ` 也不是 `^key:` 的行会走
   "折行续接"分支，**被吞进上一个键的最后一个值**。第一版因此污染了 `scope` 的第 9 项。
   ⇒ 块内一律只用 `key: value`；列表项写成 `- 轴=值`。
2. **写 frontmatter 时漏写闭合 `---`**：直接把 21 个权威文件改坏、工作区闸口从 59 变成 0。
   已回滚（git + 自制备份双保险），并给脚本加**写后自检**（fence 数量、正文首行仍在、
   落盘后复核），改为**先单文件验证再铺开**。

⚠️ **同时暴露一处方法错误并已纠正**：我用 `git status --short … | head -3` **截断了输出**，
误判"knobs/patches 无既存改动"，于是 `git checkout` 打回了一处**未提交的 09-21 更正**
（`vllm-fused-moe-tile-seeds.md` 的 device_name 段）。已从备份整体还原并逐字节验证。
**教训：判断"有没有既存改动"不许截断输出。**

## 2. `env → 版本` join 从 0/15 修到 15/15，并落成闸口

归一化规则 `resolve_env()`（去 `envs/` 前缀 + 取首段 + 8121 镜像特例）之后，
新增 **`scripts/scope_match.py --env-check`**：校验每条臂的 `rocm`/`torch`/`engine` 轴
与它所引用环境的轴一致。

🔑 **这是九轴里 `rocm`/`torch` 唯一可机器验证的地方**——因为 `config/env-lock/*.txt`
一个 ROCm 版本字符都不记，版本三元组只能三方拼，拼完就可能拼错。

**首跑即抓出一处真错（我自己引入的）**：`llama.cpp_gfx90a-2026.9.8` 一棵树里含**多个 commit
子前缀**（`qwen4exp-mtp-rp`@a9e9c3c5f、`glm5next-rp`@629b50552），而我把 env 条目的 engine 轴
写成单值 ⇒ 8108 被误判冲突。改为多值并在 `applies_to_note` 记这条事实。
修的过程中又踩一次：比较用 `str()` 相等 ⇒ list vs str 造出 4 处**假冲突**，改为复用 `_hit()`。

## 3. 新增闸口：markdown 侧的 `applies_to` 也要在词表内

第一版只验 `data/*.json`，漏了权威源。补上后立刻抓到并发会话新建配方里的越表值
（`engine=vllm-master`、`arch=Qwen4Exp`、`model=Qwen3.8-Flash-Next` —— 轴名对、值拼错，
**这类错不报错，只会让匹配永远返回「不适用」**）。已修正为词表值。
另修正一次**位置错误**：该检查最初放在 `if mid not in entries: continue` 之后，
对新配方永不生效。

## 4. Top-10 资产收编

新建 `references/60-method-measurement-gates.md`（测量有效性 / 判据设计 / 闸口自身缺陷 /
跨模型外推禁令 / 制度八条 / 归档三判据 / 报数协议六条 / 一条完整撤回案例），
并向其余 5 个 reference 追加：

| 文件 | 追加内容 |
|---|---|
| `00-T0-host.md` | `hipIpcOpenMemHandle` 必须 `flag=1`、**跨 GPU 无 happens-before**、功率模组级 + 350 W 全程通过、`GPU_DIE_NUMA_MAP` 与 tp8 无法 per-rank 绑、`.so.7`/`.so.6` 绝不混 `LD_LIBRARY_PATH`、**`-c` 是总池不是每槽** |
| `10-version-matrix.md` | **ROCm 10 整代**适用边界表、−60% 根因（RCCL 地板 113→294 µs，且 md5 一致 ⇒ 慢不是错）、5 个 RCCL 开关全无效、打包结构断裂、两个 API 破坏点、Worker 无视 SIGTERM、**ISA 断言的可执行证明协议**（MFMA 助记符改名 / gfx950=0x4f / **无对照组的"不支持"等于零信息**） |
| `20-model-quant-matrix.md` | 版本绑定核实、自转 int4 的 12 小节硬约束（TileLang mHC 静默算错、`shard_offset` 未打包单位、占比外推是错的、KV 890 vs 3660 B/token、三堵物理墙…）、**Marlin 符号 0 命中**的二进制判据、选型铁律 4 条、**indexer：bf16 承载比 fp8 存储快 2~3×** |
| `40-knobs-and-patches.md` | **真正的锁是 `block_size` 不是 head_dim**、`case 256` NO-GO、别再把 GLM 当 custom PA 靶子、MMVQ「9 行悬崖」+ **其两条自带适用域限定**、Q8_0 走 dequant 白名单、`CTX × UBATCH` 乘积约束、8103 KV 实账 6.6 KiB/tok |
| `50-dead-routes.md` | 新增 17 条判负路线（CUSTOM allreduce、分层 allreduce、`NCCL_SYMM_MEM`、patchkit `SUPERSEDED` 死代码、`-fa on` 前提…）+ 元教训「**判据一律要问：它在什么输入下必须变红？答不出就是恒绿**」 |

**收编前逐条抽查原档**：6 条最吃重的判语全部对上；其中一条发现原档自带**两条适用域限定**
（"9 行悬崖"只是 `np` 维、且只在 Q8_0 成立），已一并写入——这正是分层要防的误用。

## 5. 并发会话的实时印证与一次现场协作

本轮期间另一会话提交 `def33b9`（13:44），**把我尚未提交的 21 条回写与 TEMPLATE 改动一并提交了**
（改动完好）。它还新建 2 条配方，其中：

- `knobs/vllm-prefix-reuse-measurement.md`——**已采纳我写进模板的 `applies_preset` + `- 轴=值` 写法**
  ⇒ 模板生效的直接证据；
- `patches/qwen4exp-mtp-prefix-reuse.md`——缺 5 个必填键（曾把工作区闸口从 59 顶到 63），
  且 `applies_to` 值越表。

两条都已按"markdown 为权威源"的方向处理：**合成进 `data/*.json`**、补必填键、修越表值，
并把其内容折进方法论层。工作区闸口 **59 → 58**（补 `files:` 时顺带关掉一条覆盖率发现）。

其中 `qwen4exp-mtp-prefix-reuse` 是一条**完整撤回案例**：09-18 据"命中 87%、TTFT 降 5 倍"
把补丁设为默认，09-21 三 boot 配对复测推翻——**误因正是"冷 boot 首个请求撞 Triton JIT"**。
判语沉淀为：**警告消失 ≠ 行为改变**。

## 6. 顺带查实并更正的一处过时说法

技能原写 `$AI/docs/recipes/**` 「**不在 git**」——**不准确**：55 个配方里 **51 个入库**。
未入库的恰好只有 4 个最新文件，**也正是本轮查出从未进 `data/*.json` 的那 4 条**
⇒ 那次停摆既无 JSON 兜底也无 git 兜底。两处缺口是同一批文件，这一点比"没入库"本身更重要。

## 7. 第二轮结束时的闸口状态

```
audit_skill_recipes --dir environments|patches|knobs|ops|serving   全绿
audit_skill_recipes --scope                                        全绿
scope_match --orphans                                              全绿（0 矛盾 / 8 未记录）
scope_match --env-check                                            15/15 一致
零丢失验证                                                          495/495 行 + 33/33 标识符
$AI/tools/audit_recipes.py                                         58 项（基线 59，未因本次增加）
```

`references/` 现 6 个文件 ~1300 行；SKILL.md 530 行 / 37 KB（原 726 行 / 58 KB）。

---

# 第四轮（14:10–14:40）：裁决 8116、补齐 harvest、修一次自己的过度归因

## 1. `8116` 的"数字矛盾"裁决：**不是矛盾，是两个轴同时变了却没记**

前三轮一直挂着一条"未决矛盾"：`ports.conf` 记 38.17 t/s，技能第四轮记 68.83 t/s。
查 `quark-int8/RESULT.md` §7 的原始表格后结案——**两个数分属不同的引擎与不同的 MTP 深度**：

| 数字 | 引擎 | MTP | 口径 |
|---|---|---|---|
| 38.17 | **nightly docker 0.29.1rc1** | **1** | 加盐混合负载 |
| 41.59 | 原生 vLLM 0.28.0 | 1 | 同上 |
| **68.83** | **原生 vLLM 0.28.0** | **5** | 同上（5 次中位数） |

真正的缺陷是 **`config/ports.conf:65` 把 nightly 的数字挂在了 0.28 的引擎描述上**，
而这个误标又 propagat 进了技能与臂表。已**就地更正 `ports.conf` 注释**（带出处与三个必填轴）。

⇒ 这是"数字不带 scope"的**教科书案例**：一个数字被登记进"权威表"时丢了引擎版本与 MTP 深度，
三年后没人能判断它是不是矛盾。**分层要防的就是这种事，而这一次它抓到了自己的上游。**

⚠️ **并且我在写归因链时先写过强了**：我照 `RESULT.md` 的「完整增益」段写了
"环境切换 38.17→41.59（+8.9%）"，但同文 §5 L274-277 明确**这一步不是单变量**
（env + `MI250_GATE_GEMV` + `--disable-custom-all-reduce` + splitKV 同时变）
⇒ **不得读成「原生 env 更快」的因果**。三处（`ports.conf` / SKILL.md §4 / `references/30`）
已同步补上该限定。**这是本轮我自己犯的错，由复核原档抓出。**

## 2. 负载标签纪律（**修正了 σ≈4.6% 那条规则的适用条件**）

`RESULT.md` §7ter 用"固定 prompt + 丢弃首次"把臂内离散从 **12–24% 压到 0.1%**，
并给出结论：**那 12–24% 几乎全是内容变化，不是计时噪声**。推论三条，已写进 SKILL.md §1.5：

- 项目的 **1.036× 噪声门只在"固定 prompt"口径下成立**；
- 加盐中位数只能定到 ±10%，**检不出 5% 级效应**；
- 单流数字必须带**负载标签**：同一臂同一配置，
  结构化可预测 ~86 / 加盐混合 ~68.83 / 开放分析型 ~56。

另把 **第三类假结论来源**补进 `references/60` §14：同机并行会话的污染。
两个实录——「`mamba_cache_mode=align` 代价 −23.7%」是假的（征兆：`steps + accepted ≠ tokens`），
真值 ≈0（spread 0.0%）；CPU 侧 `-t 48 --poll 0` 方差比均值还大。
纪律归纳成三问：**预热同形状了吗？这台机器上有别人的负载吗？步时分解自洽吗？**

## 3. harvest 补齐并**量化覆盖度**

建了一个可复跑的覆盖度度量：对 41 个 harvest 小节，各取其**独有硬 token**
（反引号内容 / 文件路径 / env 名），看在 `SKILL.md ∪ references/ ∪ data/` 里的命中率
（≥60% 记为已收编）。结果：

```
HARVEST 覆盖：✅32  🟡8  🔴0  / 共 41
```

唯一的 🔴（H35 `davetha-wu1w-gfx90a-落地施工单`）经核是 **token 形式假阴性**——
实质内容早已在 `references/60` §4，真缺的只是权重来源坐标 `Freaksterz/…W8A8-INT8`，已补。
8 条 🟡 是**有意的**："结论已落库、逐条行号明细留在 `.tmp/harvest/docs-top.md`"——
按 §7.1 的纪律，技能里只留结论 + 指针，不把 41 小节全文搬进来撑爆常驻上下文。

本轮新收编的高价值条目（均**逐条回原档复核**，6/6 命中）：

| 层 | 内容 |
|---|---|
| T1 版本 | aiter 包分 `aiter/` 与 **`aiter_meta/`** 两层 ⇒ **只 grep 前者会把"wheel 没 ship"误读成"上游没有"**；`module_gemm_a8w8` 的 JIT 在**引擎初始化路径内**（8 rank 串行等锁 ⇒ 首次起服 **2310 s**）；RMSNorm 雷的精确坐标 `platforms/rocm.py:1109-1116`（**sitecustomize 不管这颗雷**）；`--check` 与 `--require` 必须分开 |
| T2 模型×量化 | **四个静默数值损坏机理**：int4≠fp4、CK stage2 的 `is_bf16_atomic_supported()=false` + **只在 `KBatch>1` 才报错**、**atom 把 `I8/U8` 无条件映射成 `fp4`**、123 个 root-only 权重文件（加载 200 s 后才崩）；「**内核存在 ≠ 在该形状可用**」的定价法与两个排除污染的核对法；MXFP4 判死措辞改准（两条独立理由）；**权重带宽下界算法**（2% < 噪声门 ⇒ 该路不通，对照 int8 Linear +14.9% 大一个数量级） |
| T0 主机 | BF16 只能 TP8（TP4 装不下 + PP 被限 1）；KFD `ulimit -l` 触发 SVM 死亡螺旋 |
| T0 方法 | KFD SVM 死亡螺旋的**完整签名 + 四条已排除**；`pgrep VLLM` 漏 `VLLM::Worker_TP0`；D 态 worker 显存不还；`diskstats $7` 是读耗时不是扇区；`tee` 吃掉退出码；**拓扑对 AIS 无影响**；**swap 必须与模型盘分离**（0.90→1.91 GB/s，且"零换页"不等于"不花钱"）；判盘必须加 4K 随机 O_DIRECT；`ledger.py gate` **不因 `superseded` 放行**；CPU 线程数悬崖与 `-fa on` 在 CPU 无效 |
| 开关 | `mamba_cache_mode` 三档语义表 + `all` 的内存几何（∝ `max_model_len/mamba_block_size` ≈481 块，**固有代价不是 bug**）；AITER ASM attention「**有库、无调用方**」；ATOM 墙 1 的编译单元连带机理 + `VLLM_PLUGINS=` 必需 |
| 机制 | **那条 9 行警告为什么会出现**：vLLM 两条标注规则 qwen4exp 都不中，而草稿层形状与规则 2 完全一致、**只是模型名不匹配** ⇒ 兜底分支把所有组当草稿组 ⇒ 这从机制上解释了更正 3 的「警告消失 ≠ 行为改变」。**通用判语：按名字设的门，换个模型名就可能误开/误关；先读门条件的输入是什么。** |

## 4. 第四轮结束时

```
audit_skill_recipes --dir ×5 + --scope           全绿
scope_match --env-check / --orphans              全绿
零丢失验证                                        全绿
工作区配方闸口                                     58（基线 59）
SKILL.md                                         536 行 / 41.5 KB（原 726 / 58 KB）
references/                                      7 文件 1666 行
harvest 覆盖                                      ✅32 🟡8 🔴0
```

`ports.conf` 的更正改动在 `/home/qiba/ai` 仓（未提交，属该仓工作流）。

## 5. 最终验收时自查出的第 5 个错（我自己的工具里的静默 bug）

验收要演示"分层能否真的回答跨 scope 问题"，跑 `scope_match.py --arm 8114… --kind patch`
得到 **0 / 0 / 0**。若我当时照着手写的结论文案交差，就会把"过滤器匹配不到"当成
"没有任何补丁适用/不适用"——**正是本任务反对的那类未验证断言**。

**根因**：`kind` 用 `key[:-1]` 猜单数，而 `arms/knobs/ops` 恰好规则、
**`"patches"[:-1]` == `"patche"`** ⇒ `--kind patch` 从第二轮起就**静默返回空**。
⇒ 改为显式 `KIND_SINGULAR` 映射 + `--kind` 单复数双向容错，并加回归
（`--kind patch` 与 `--kind patches` 结果必须一致）。

**修好后的实证**：`vllm-dsa-indexer` 对 8114 判 **❌ 不适用**，
否决轴 `arch: 需要 ['deepseek_v32','glm_moe_dsa','deepseek_v41']，目标 ['qwen3_5']`。
⇒ 场景「把 8121 的 `DSV41_IDX_AITER_KERNEL=1` 搬到 8114」得到一条**可执行的否决**，
而不是"靠人记得那是 GLM/DSV 的旋钮"。**这是整套分层设计的验收标准，现在它真的能用了。**

## 6. 五处自查出的自身错误（全部留痕，按仓库纪律不静默改写）

| # | 错误 | 性质 |
|---|---|---|
| 1 | 写 frontmatter 漏闭合 `---`，改坏 21 个权威文件 | 破坏性；已双备份回滚 + 加写后自检 |
| 2 | `git status \| head -3` 截断致误判，`git checkout` 打回他人未提交更正 | 判断依据不完整；已还原并逐字节验证 |
| 3 | markdown 侧词表校验放在 `continue` 之后，对新配方永不生效 | 检查位置错误 |
| 4 | 把 `38.17→41.59` 写成"环境切换 +8.9%"的因果，未查它不是单变量 | **过度归因**；三处补限定 |
| 5 | `--kind patch` 因 `key[:-1]` 猜单数而静默返回空 | 工具 bug，被"拒绝未验证断言"的验收习惯抓到 |

---

# 第五轮：1–5 全做（含一次并发覆盖的检测与合并修复）

## 0. 本轮最重要的事：**分层工作被并发会话整段覆盖，被发现并合并修复**

`audit_skill_recipes.py --dir serving` 突然报「8121 没有 JSON 条目」+「缺 applies_to」。
逐文件取证结果：并发会话 15:51 / 15:55 两笔提交（其真实目的是 KB recipe v9/v10）
**基于一份过期快照重写了五个 `data/*.json`** ⇒ 15 条 `applies_to` 退回旧 `scope`、
6 个条目消失。这正是我在 §7.1 写下、并被并发会话自己确认过的失效模式，**这次发生在自己身上**。

修复方式是**按文件粒度合并**而不是回滚：
- 逐字段比对 `7955ee0 → 15ddd65`，确认他们的实质内容只有 8107 一条，
  且**是我的严格子集**（`verified` 退回 09-15，少 3 metric + 2 pitfall）；
- 确认他们的 DCP lessons 在 `hyperloom/kb/.../recipe.json`（独立文件，不受影响）；
- 于是恢复五个 data 文件到我的最后好版本、**保留他们的 `recipe_kb.json`** 只补回 `scope_spec` 指针、
  重放本轮对 data 的两处更正、重加 4 条新臂。
- ⚠️ 中途我还差点反向覆盖他们：`不是一套门` 那段文本在恢复后"消失"，追查发现它在
  `hard_rules` 而非我扫描的 `decision_rules` —— **三方版本里都在**，是我的检索字段搞错，非覆盖。

**教训（已写进 §7.1）**：并发环境里 `git checkout <file>` 前必须逐字段 diff 双向，
"我恢复我的"和"我删了他们的"只差一次字段定位。

## 1. #4a：装 a8w8 表——**并推翻本技能对它的定性**

装之前先查了运行时到底怎么用这张表，结果是本轮最有价值的发现：
`aiter/ops/gemm_op_a8w8.py:656-663` 显示该路径**只消费 `splitK` 一个字段**，取不到就 `splitK=0`；
而 54 行 gfx90a 的 `splitK` **全 = 0** ⇒ **行为完全等价**。
再看 `patches/gfx90a/README.md:20`——工作区早就写着「**只为消掉每次启动 432 行 `not found tuned config`，
性能恒等**」。

⇒ **本技能的 §9 把它写成"尚未回收的便宜杠杆"是错的**，而且
`data/knobs.json::vllm-fused-moe-tile-seeds.hard_rules` 一直写着「不要去"补表"」——
**技能内部自相矛盾，我照错的半边执行才回查出来**。两处均已留痕更正
（`hard_rules` 补充：判词的准确含义是"别指望收益"，而非"补了会坏"）。

过程中还差点酿成事故：`build_tune_dict` 是 **fail-loud** 的（kernelName 不在 registry 就
`RuntimeError` 并拒绝产出 `.so`），而这 54 行的 `kernelId=-1 / kernelName=ck_a8w8_auto`
在三个 env 的 aiter 里**都搜不到**。查明它只作用于 AOT/JIT codegen 路径（本仓用预编 `.so`）、
运行时是纯 pandas 查表，才确认可行。**若直接追加就会打挂引擎初始化**。
另外还查了 `links=1` + inode 各异（三个 env 是 `cp -a` 出来的，硬链接会让"逐 env 改"语义失效）。

装后独立复核：三 env 各 580→634 行、54 条 gfx90a、表头完好、与源表同集合、pandas 可解析、
`splitK` 唯一值 `[0]`、54 个唯一键；备份 `.bak-20260921` × 3 + 一行回滚命令。
**全程未起任何服务，8 张卡始终空闲。**

## 2. #4b：四条臂补齐 serving 配方（19/19 可解析）

`8115 / 8116 / 8117 / 8119` 此前只有 launcher 和端口登记，**没有配方** ⇒ 进不了 `data/arms.json`。
四条配方全部按实测写（defaults 逐条从 launcher 抽取，闸口的 `defaults:` 反查零发现），
并合成进 arms.json（19 条，全部带 `applies_to`）。三条实测纠正：

- 🔴 **8119 的默认权重是悬空软链**（`…-engram4` 真盘不存在，只有 `-opt-allbf16`/`-pilot`），
  `:183` fail-closed 会拒 ⇒ **我第一轮把它记成 active 是错的**。
  契约状态集 `{active,archived,dead,experimental}` **没有 `blocked` 档**（闸口直接拒绝该值），
  我没有擅自改共享契约，改为 `experimental` + 标题/why/§1 三处显式警示，
  并**建议给 STATUSES 增加 `blocked`**——"路线有效但实物缺失"是真实且会复发的状态。
- ⚠️ **8115 文件名说 `int8w8a8`、默认权重却是 `-Attn` 变体**；**8116 文件名说 `mtp1`、默认却 `SPEC=5`**。
- ⚠️ `…_OFFICIALENC_mxfp4_8119` 名字写 mxfp4、**内容其实是 CT-Int4** ⇒ 别按文件名判量化形态。

## 3. #4c：`audit_log_paths.py`——**我的 todo 是过期的，但真问题在别处**

实测：`/tmp` 写点漏检**早在 09-16 已修**，全仓命中 0。我照报告写 todo 而没先验证，是错的。
但顺着查发现**同一类缺陷仍在**：PID 消费方检查**没豁免整行注释** ⇒
**19 条发现全是假阳性**（launcher 头注释里的 `# kill -TERM -$(cat logs/ornith397b-8115.pid)`
被当成"按把手找"，而该脚本真正的 `PID_FILE` 用的是规范式）。
补注释豁免 + 登记 VOCAB（8115/8116/8117/8127/8119/8121）后：**19 → 4**，
剩 4 条是真的（8121 无 MODEL_KEY；三条 8119 用 `%H%M%S` 违反 `%H%M`——
那其实是**同一个分钟级撞名隐患的反向解法**，改动需拍板，我未擅自动 launcher）。
新增整行注释豁免前先用 11 个用例正反测（含"行尾带注释的写点仍要报"）。

## 4. #5：把口径纪律变成可跑的检查

新增 `scope_match.py --metric-audit`：吞吐数字必须带负载标签、含因果/前后对比表述必须声明是否单变量。
**第一版报 56 条、其中大半是误报**（把 `KV 池`、`步长分解`、`归属` 当吞吐）。
我按自己写的纪律把它砍到 **27 条真该补的**——因为"一个天天误报的闸口会训练人忽略它"。
两轮收紧：① 只管"是速率且非容量/归属"；② 扩充负载标签词表（`filler`/`自然内容`/`短 ctx`…）。

## 5. #3：开集轴语义显式化

`--scope` 现在把 `model` 轴印成
`词表 1 项（**开集轴：只登记需特别标注的类别，不穷举取值**） 实际用到 16`，
不再会被误读成"覆盖不足"（我自己第一轮就误读过）。

## 6. 验证器也补了一类判定

行级零丢失验证会把**有意更正**误报成丢失（本轮 2 行：`便宜杠杆` 标题与结论）。
加了 `CORRECTED` 白名单：旧行允许消失，但**必须证明替代文本已就位**——
否则"修正错误结论"会被自己的验收工具判成失败。

## 7. 本轮终态

```
audit_skill_recipes --dir ×5 + --scope            全绿
scope_match --orphans / --env-check               全绿（19/19 臂 env 可解析）
scope_match --metric-audit                        27 条 advisory（吞吐缺负载标签）
零丢失验证                                         495/495 逐字 + 2 处有意更正在位 + 33/33 标识符
audit_log_paths（工作区）                           19 → 4 条发现
audit_recipes（工作区）                             58 → 52 条发现
GPU 占用                                           无（未起任何服务，8 卡空闲）
```
