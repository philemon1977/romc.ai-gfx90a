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
