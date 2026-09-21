# T3 · 逐臂细节（臂级事实，换任一轴即失效）

> **T3 组合层** —— 每条都绑定 (engine版本 × rocm × torch × 模型 × 量化 × 拓扑) 的完整组合。
> 机器可读版在 `data/arms.json`（带 `applies_to`）；本文件放**放不下 JSON 的上下文与口径警告**。
> 用 `python3 /home/qiba/ROCm.AI/scripts/scope_match.py --arm <id>` 先判适用性，再来读细节。

## 8107 · Qwen3.8-Flash-Next 176B BF16 · vLLM master TP8

**scope**：`engine=vllm-master@8a728663c` · `rocm=host-7.2.4` · `torch=2.12.0+git6bbd260` ·
`arch=qwen4exp` · `quant=bf16` · `topology=tp8`

### 数字口径（**同一臂三个时点，不可互相顶替**）

| 时点 | 单流 | Phase2 聚合 | 口径 |
|---|---|---|---|
| 2026-09-05 | 88.6 | **296.86** | 560 W、SPEC=3、batched 8192 |
| 覆盖期 | 92.07 / 93.16 | — | 同上 |
| 合并 8 条臂级 env 后 | **94.85 / 94.62 / 95.29** | — | acceptance **89.9%**、step **38.9–39.2 ms** |

- **那 8 条 env 性能中性**；KV 池 **719,056 tok（2.74×）**。
- ⚠️ `VLLM_ENABLE_V1_MULTIPROCESSING=0` 在 vLLM master 这版**并没有**把 EngineCore 并回
  APIServer（日志里仍有独立 `(EngineCore pid=…)` 行）⇒ **别按"应该同进程"去推断进程拓扑**。
- 🔑 单流 88.6 → 94.85 这一档差异（~7%）**大于**跨 boot 噪声 σ≈4.6%，但**同 boot 内**的
  三次重复（94.85/94.62/95.29）spread 仅 0.7% ⇒ 引用时必须说明取的是哪一时点。

### 起停与文件

- 停服**三步法**与组杀的 PGID 边界见 `references/00-T0-host.md` 附录（本臂 TERM 会挂住，
  worker 报 `FileNotFoundError: /psm_*`）。
- **同一条臂有 3 份实体**：基脚本 / `launcher/favor/…_final_….sh` 薄封装 / `/mnt` 镜像。
  改动只落基脚本 + `md5sum` 对账 + 侧车软链。
- 日志已改落 `logs/flash-next/server-<port>-<ts>.log`（+ `.current` 软链）⇒
  **取日志读 `.logpath` 侧车，别用旧的 glob**（会静默读到旧日志）。
- ⚠️ 8107 上挂**两个引擎**（vLLM BF16 占 8 die；llama.cpp GGUF 占 die0-3），
  **同 PID 文件、同端口 ⇒ 同刻只能起一个**（`config/ports.conf` 已声明互斥）。

### 与补丁的关系

- `patches/gfx90a/port_qwen4exp_eagle_annotate.py`（env 门 `VLLM_QWEN4EXP_EAGLE_ANNOTATE=1`）
  对本臂**吞吐中性**，只消 9 行误导性警告 ⇒ 见 `references/60-…` §9 与 `data/patches.json`。
- `qwen4exp` 这条 arch **只有 `envs/vllm_master_rocm724` 注册**（0.28 裸轮 grep 得 0）
  ⇒ 换 env 不是调参问题。
