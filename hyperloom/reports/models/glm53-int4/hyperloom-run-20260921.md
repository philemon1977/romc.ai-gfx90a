# Hyperloom optimize 在 GLM-5.3 CT-INT4 上的启动记录（2026-09-21）

**会话目录**：`hyperloom/session/GLM-5.3-CT-Int4-W4A16/20260920T185444Z-4735e5aa`（kernel-agent 侧）
**运行日志**：`hyperloom/session/optimizer_runs/run_GLM-5.3-CT-Int4-W4A16-20260920_185420.log`
**配置**：TP=8 EP=1 CONC=32 ISL=OSL=1024 PRECISION=w4a16 MAX_HOURS=12 TARGET_GAIN=50%
**server-args**：`--max-num-batched-tokens 2048 --max-num-seqs 32 --max-cudagraph-capture-size 8
                  --decode-context-parallel-size 8 --gpu-memory-utilization 0.95`
**flags**：`--gpu-type mi300x --max-model-len 32768 --max-minutes-framework-pct 0.43 --max-minutes-kernel-pct 0.42`

## 启动前修掉的四个坑（都不是 Hyperloom 的错，是本机环境）

1. **agent 的 LLM 端点过时**：`.env` 原指 `http://127.0.0.1:9100`（本地网关，未运行，工作区无启动脚本）。
   改为 `https://api.deepseek.com/anthropic`（实测 HTTP 200），`CLAUDE_MODEL` 由用户定为 `deepseek-flash`（实测 200）。
   ⇒ agent 的 LLM 走远程，**不再需要本地卡**，原先"agent 模型与优化目标抢卡"的死结消失。
2. **install.sh 卡在 github 拉取**：容器出网对大流量只有 ~45 KB/s（ls-remote 秒回、clone 却爬）。
   先把 TraceLens / GEAK / InferenceX 的**既有检出**（都在钉住的提交上）拷成容器内镜像 `/opt/gitmirror/*`，
   再用 `git config --global url.<本地路径>.insteadOf <github url>` 把 fetch 变成本地瞬时（实测 5 s）。
   另外 metrix 依赖的 `AMDResearch/intellikit` 在宿主克隆（<90 s）后同样做成镜像（含钉住的 `2f61453a`）。
   ⇒ `install.sh` 最终 **INSTALL_RC=0**，Magpie #C1 补丁也打上了（`atomic_ok=True`）。
3. **`.env` 里三个解释器路径是 baremetal 时代的残留**：`PYTHON`/`MAGPIE_PYTHON`/`MAGPIE_PATH`
   指向 `/opt/envs/vllm/...`、`/opt/venv/...`（容器里都不存在）⇒ 预检直接 `RC=127`。
   改为 `/usr/bin/python3` 与 `/usr/local/lib/python3.12/dist-packages`（容器里 vllm 0.3.1.dev85 + torch 2.12 都在这）。
4. **skill 的 `launch.sh` 有引号 bug**：`${OPT_FLAGS}` 无引号展开，实测 `--a "--b c d"` 会被拆成
   `["--b] [c] [d"]` ⇒ `--server-args`（值含空格）必然传坏。本仓改用 `quark-int8/scripts_local/hl_launch.sh`，
   用数组传参，其余 env 链/setsid/日志/launch-info/last_launch.env 与 skill 逐字一致。

## IR-1 的授权例外（留痕）

预检默认要求每个 GCD < 500 MiB。当时另一个会话的 `probe_int8_gemm.py` 占着一个 GCD 的 503 MiB，
用户明确授权"可以抢卡，没有其它任务在跑"。我**没有杀那个进程**，而是：
① 显式设 `IR1_VRAM_LIMIT_MIB=1024` 通过门；② 在 server-args 里加 `--gpu-memory-utilization 0.95` 留出 ≥1.3 GiB 余量。
（那个探针不是 Hyperloom、也不是本会话起的。）

## 已知的"参考而非判据"之处

- `--gpu-type mi300x` 是**显式声明**：本机 gfx90a 不在 Hyperloom 的探测表（只认 gfx942/gfx950），探测返回 None 时采用声明值。
  ⇒ 它算的 roofline 是 **MI300X 口径**（带宽约为 MI250X 的 3 倍），结论对本机只能作参考。
- 12 小时期间 8 张卡由 Hyperloom 起的 vLLM 服务占用。
