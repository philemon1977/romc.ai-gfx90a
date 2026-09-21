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


