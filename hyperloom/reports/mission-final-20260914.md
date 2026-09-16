# ROCm.AI 任务最终报告 — 2026-09-14（gfx90a 三步计划）

机器：8× MI250X (gfx90a/CDNA2)，Ubuntu 24.04，amdgpu 6.16.13。
模型：`/mnt/stripe-3mix-3t2/models/davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8`（W8A8 INT8 compressed-tensors，Qwen3_5 GDN 混合架构，head_dim=256，29.1GB 权重）。

## 结论一览

| # | 任务 | 结果 | 状态 |
|---|------|------|------|
| 1 | Hyperloom INT8 会话增益 | `enablement_stalled`，baseline 0.0，增益 0.00%（未验证） | ❌ 未出数，归因完整 |
| 2 | AITER FA 移植 + A/B | conc32：40.50 vs 40.89 tok/s（+1.4%，噪声内）→ 负结论 | ✅ 有定论 |
| 3 | 8114 生产 launcher 验收 | 三条 grep 全过；单流 49.9 tok/s（文档 80.3）→ 机制正常，差距归因输出风格 | ✅ 有定论 |

## 1. Hyperloom 会话（失败归因）

- 最终报告：`session/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8/20260914T134902Z-244d9455/reports/final.md`（final.json 为 root:600）
- 包归档：容器内路径 `/workspace/hyperloom-session-packages/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8_20260914T134902Z_5ae731fb.zip`
- 3h 预算 100% 消耗于 PRELUDE enablement 循环（robustness 告警：PRELUDE 预算 1129%）。四类基础设施故障全部定位且有证据：
  1. **HIP/ROCR 设备掩码矛盾**：harness 泄漏物理 HIP 掩码 + Magpie 叠加逻辑 ROCR → torch 严格校验崩溃或"无卡可用"。会话内 specialist 已写出 `_reconcile_rocm_device_masks` 补丁并在沙盒 E2E 证明（权重加载、35/35 graph、/health 200），但 integration lane 直到硬截止都没有完成应用（policy_denied `enablement_round_in_flight` ×9 拖慢节奏）。
  2. **bench client 解释器缺依赖**：yaml `PATH` 首位 `/opt/venv/bin` 无 transformers/huggingface_hub；会话从 pypi.org 安装被连接重置（本机屏蔽）。已由外部以 aliyun 镜像代装验证 import OK（16:19），但为时已晚。
  3. **端口冲突史**：host unsloth studio 占 8888（已按授权清除，可用 `~/ai/start-unsloth-studio.sh` 复原）。
  4. **幻影租约/resume 脱同步**：kill+resume 诱发 coordinator.db 幽灵租约 → IR-1 门死锁；恢复配方已验证（tasks→failed、leases 回填过期）。另发现 orchestrator turn 上限 12 导致对话式角色停滞，已本地改 36（`hyperloom/orchestrator/roles/claude.py:132`）。
- gfx90a 不在 Hyperloom 支持矩阵（MI300X/325X/355X），是上述摩擦的底层原因。
- 建议后续：向 AMD-AGI/Hyperloom 提 issue（enablement 门 + HIP/ROCR + turn 恢复死锁）；或直接手工基线（见 §3 已有真实数字）。

## 2. AITER FA 移植（gfx90a）

- 移植产物：`hyperloom/envs/vllm-fa/`（13G 环境副本，`aiter_meta/hsa/gfx90a/` 安装、aiter+vLLM 双闸门打开、`module_fmha_v3_fwd.so` 重建）；脚本 `hyperloom/scripts-local/fa_port.sh`、`fa_ab.sh`；日志 `hyperloom/logs/fa_ab*.log`。
- A/B（GPU7、INT8 模型、conc8/32、4096 长 prompt）：arm2（FA on）40.50 vs arm1 40.89 tok/s @conc32 → **无增益**。
- 根因：本模型 head_dim=256 被 AITER FA decoder 路径拒绝，AITER FA 仅命中 ViT/MMEncoder；gfx90a ASM FA 可移植上限 = hd128 / 192×128（242/1422 kernel）。
- 资产价值：移植流水线可复用于 hd≤128 模型（Qwen3 dense 原版、多数 8B/14B）。

## 3. 8114 生产验收（真实吞吐数字在此）

- 栈：wu1w env（`/opt/envs/wu1w-int8-028`）+ `~/.local` user-site shim + `vllm-wu1w` 包装；`--quantization compressed-tensors --language-model-only --cudagraph_mode FULL_DECODE_ONLY --speculative-config dflash`。
- 验收：三条 grep 全过 — 平均 12.0 draft tokens/草稿（dflash 工作正常）、tokens/step=2.66、ms/step≈52。
- 单流自然散文 **49.9 tok/s** vs 文档 80.3：机制无故障；接受率 13.8% vs 文档 21%，差距源于该模型 reasoning-echo 输出风格（"We need to answer…"复读）压低单步消耗——与文档 §5.1 教训同类。
- 日志：`/home/qiba/ai/logs/qwen38-27b-int8/server-8114-*.log`（含 current 指针）。服务已按用户操作停止；重启命令见 `~/ai` 内对应 launch 记录。

## 遗留与环境状态

- `hyperloom-srv` / `hyperloom-fa` 已 stop（未删，`docker start` 可复；srv 重建后需重放 Magpie sed + `/usr/local/bin/vllm-wu1w` shim）。
- 容器内 root 属主的 `final.json`（600）需 sudo 读取。
- ROCm 四环境（pytorch/tf/jax/vllm）+ AMD Skills（8 个，三处个人级软链）此前已全部验证 PASS，见顶层 `README.md`。
