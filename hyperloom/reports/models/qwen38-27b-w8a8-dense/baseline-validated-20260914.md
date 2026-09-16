# `Qwen3.8-27B-ABLITERATED-W8A8-gdnint8`（dense INT8 W8A8）— baseline 打通

- 模型：`/mnt/stripe-3mix-3t2/models/davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8`
- 架构：`Qwen3_5ForConditionalGeneration`，**dense**（config 里无 `num_experts` / MoE 键）
- 权重：**27.9 GB**，2 分片；量化 `compressed-tensors`，W8A8 int8（per-channel 权重 + token 动态激活），`lm_head` 亦在量化目标内
- 特殊结构：**混合注意力**（多层的 `linear_attn.in_proj_a/b`）＋ **MTP 层**（`mtp.*`，8 个，未量化）
- Multimodal：config 带 `vision_config`（`Qwen3_5ForConditionalGeneration`），但本仓库一律跑**纯文本路径**（Hyperloom 报 `DEGRADED MODE`）；`model.visual.*` 在量化 `ignore` 列表里
- 机器：8× MI250X (gfx90a/CDNA2)，ROCm 7.2.4，vllm 0.28.0+rocm723

## 结果：`enablement_stalled` + `baseline 0.0` 已消除

Session `Qwen3.8-27B-ABLITERATED-W8A8-gdnint8/20260914T173139Z-3ae15c89`，预算 120 min。

| 项 | 本次 | 上一次（同模型/机器/负载） |
|---|---|---|
| stop_reason | `time_exhausted` | `enablement_stalled` |
| baseline | **444.9 tok/s/GPU** | `0.0` |
| current_best | 444.9 tok/s/GPU (`action=baseline`) | `{}` |
| cumulative_gain_val | `0.00%`（预算内未落地 explore KEEP） | `0.00%` never validated |
| crash_count | 0 | 0 |

Server：`Application startup complete` + `GET /health 200 OK`，启动 **180.8 s**，`ROCR_VISIBLE_DEVICES: '0'`。
精度门：gsm8k `exact_match,strict-match = 0.9674`（1319 生成，10 条截断）→ 该 baseline 为**吞吐+精度双通过**。

### 吞吐全量（`artifacts/inferencex_result.json`）

| 指标 | 值 |
|---|---|
| `output_throughput` | **444.86 tok/s**（TP=1，单 GCD，conc 64） |
| `total_token_throughput` | 889.71 tok/s |
| `request_throughput` | 0.434 req/s |
| `total_output_tokens` / `total_input_tokens` | 327,680 / 327,680（320 prompts × 1024 OSL/ISL） |
| `duration` | 736.6 s |
| `mean / median TPOT` | 137.8 / 140.5 ms（std 6.9 ms） |
| `mean / median / p99 TTFT` | 6121 / 3349 / 33172 ms |
| prefix cache 命中率 | 51% → 21.9%（换批后） |

### 配置（`artifacts/baseline_config.with_envs.yaml`）

```
TP=1  CONC=64  ISL=1024  OSL=1024  NUM_PROMPTS=320  NUM_WARMUPS=8  MAX_MODEL_LEN=6144
ROCR_VISIBLE_DEVICES=0
VLLM_ROCM_USE_AITER=1  VLLM_ROCM_USE_AITER_LINEAR=1  AITER_LOG_TUNED_CONFIG=1
VLLM_BIN=/usr/local/bin/vllm-wu1w          # wu1w-int8-028 环境 + usersite-shim
EXTRA_VLLM_ARGS=--trust-remote-code --language-model-only --quantization compressed-tensors
                --safetensors-load-strategy eager
                --compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}
```

## 该模型上生效的三处改动

1. **Magpie reconcile 补丁**（`#1505`）：`Magpie/modes/benchmark/benchmarker.py`，2 hunk，对 1992 行原件 `git apply --check` 与 `patch --dry-run` 均干净无 fuzz。
2. **`_CONVERSATIONAL_MIN_MAX_TURNS` 12→36**（`#1506A`）：改的是**被 import 的那份** `/opt/envs/vllm/lib/python3.12/site-packages/hyperloom/orchestrator/roles/claude.py:132`。
3. **源头拆掉掩码泄漏**：launcher 里 `unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES`（容器 env 带 `HIP_VISIBLE_DEVICES=0..7`，而 TP=1 只贡献 `ROCR_VISIBLE_DEVICES:'0'`）。

**归属说明**：第 3 条使 reconcile 调用退化为惰性（无矛盾时返回 `None`），所以本轮**不能证明补丁本身被触发**，只能证明该症状类消失。两条互补，都保留。

预算参数另调两处，属 `shared/` 的编排问题，见 `../shared/orchestration-budget-and-teardown.md`。

## 未验证的余量（**仅本模型**）

本模型在 conc 64 下的每步耗时推算：

| 推算 | 值 |
|---|---|
| 聚合 444.86 tok/s ÷ 64 并发 | 6.95 步/秒 → **144 ms/步**（与 TPOT 均值 137.8 ms 吻合） |
| 每步须读权重 | **27.9 GB**（decode 访存受限，与 batch 无关） |
| 实测有效带宽 | 27.9 GB / 0.144 s ≈ **194 GB/s** |
| MI250X 单 GCD 标称峰值 | 约 1.6 TB/s → **利用率约 12%** |

即：conc 64 时它**不是权重带宽瓶颈**（batch 1→64 只换来 8.9 倍吞吐，带宽受限应接近 64 倍），有某个随 batch 线性增长的开销在主导。TPOT 标准差仅 5%，说明是稳定的每步成本。**这是本模型上最大的未解余量**，尚未定位。

注意：这条推算只对本模型的 shape/量化成立，不可外推到 MoE 模型。

## 产物

- `artifacts/inferencex_result.json`、`artifacts/baseline_config.with_envs.yaml`、`artifacts/server_ready_at`
- 原始 session：`session/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8/20260914T173139Z-3ae15c89/reports/final.md`
- launcher：`scripts-local/launch_int8_baseline.sh`
- 镜像：`rocm-ai/hyperloom:runtime-20260914-patched`
