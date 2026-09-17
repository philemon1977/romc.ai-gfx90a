# INT4 W4A16 via compressed-tensors（替代 MXFP4 emulation）— 侦察与准备报告

## 1. 为什么不能继续用 Quark 出 int4（实测证据，纯 CPU）

用同一段已验证可用的 mxfp4 配置做 A/B（同代码路径，只改 weight 字段）：

| 配置 | vLLM `QuarkConfig` 解析结果 |
|---|---|
| mxfp4 per_group 32 e8m0（v3 在用） | ✅ `QuarkOCP_MX` |
| **int4 per_group float** | ❌ `NotImplementedError: No quark compatible scheme was found` |
| uint4 per_group float | ❌ 同上 |
| int4 per_channel float | ❌ 同上 |

Quark 的 scheme 只有 `w8a8_fp8 / w8a8_int8 / ocp_mx / nvfp4 / w4a8_mxfp4_fp8`。
`_POSSIBLE_KERNELS[PlatformEnum.ROCM]` 里确有 `TritonW4A16LinearKernel`，MoE 侧
`int_wna16` oracle 的后端是 `MARLIN / BATCHED_MARLIN / TRITON`（ROCm 优先级含 TRITON），
但**消费者是 AWQ / GPTQ / compressed-tensors / moe_wna16**，没有 Quark：

```
auto_awq.py, auto_gptq.py, moe_wna16.py,
compressed_tensors/.../compressed_tensors_moe_wna16.py,
compressed_tensors/.../compressed_tensors_moe_w4a16_flydsl.py
```

→ 结论：要让 MI250X 吃到 Triton W4A16，**必须换 checkpoint 格式**（compressed-tensors），
而不是改 Quark 的 spec 字段。

## 2. 新配方：compressed-tensors `pack-quantized` int4 W4A16

- 量化：weight-only **int4 对称**、per-group（默认 group_size=128）、float32 scale（RTN，**无需校准**）
- 激活：保持 bf16（W4A16）
- 磁盘布局（每 2D 权重，compressed-tensors 约定）：
  - `<name>.weight_packed` int32 `[N, K/8]`（每个 int32 塞 8 个 int4）
  - `<name>.weight_scale` float32 `[N, K/group_size]`
  - `<name>.weight_shape` int64 `[2]`
- MoE：沿用已验证的融合专家拆分（`mlp.experts.gate_up_proj` → `experts.{i}.gate_proj/up_proj`，
  `down_proj` 同理），排除表与 v3 一致（router / shared_expert / norm / conv / lm_head / vision / mtp / linear_attn / self_attn 保持 BF16）
- config.json 写入：
  `{"quant_method":"compressed-tensors","format":"pack-quantized","config_groups":{...num_bits 4, strategy group, group_size 128...},"quantization_status":"compressed"}`

## 3. 已完成的验证（全部 CPU，无 GPU/重 IO）

| 检查 | 结果 |
|---|---|
| vLLM 解析 CT 配置 | ✅ `config parsed OK; quant_format = pack-quantized` |
| CT 自带 pack→decompress 往返 | ✅ `weight_packed int32 [64,32]` / `weight_scale fp32 [64,2]` / `weight_shape int64[2]`，normRMSE 0.1149 |
| packer 在 tiny 上运行 | ✅ 673 张量 |
| **融合专家布局样本**（无 `.weight` 后缀） | ✅ `split 8 fused expert tensors; quantized 12 2D weights` |
| int4 数值往返 vs bf16 源 | ✅ gate/up/down：normRMSE **0.117**、cosine **0.9932**（与 MXFP4 的 0.112/0.9937 同级） |
| 排除层 | ✅ 保持 BF16 |

**踩到的静默 bug（已修 + 已加断言）**：融合专家张量名 `...experts.gate_up_proj` **没有 `.weight` 后缀**，
我最初用 `core.endswith(".mlp.experts.gate_up_proj")`（`core` 已 rsplit 掉末段）判断 → 永不匹配 →
真实模型 98% 权重会被静默跳过。现改为按 `name` 判断，并在结尾断言 `quantized_tensors > 0`。

## 4. 待执行（需要 GPU / 重 IO，等你放行）

| 步骤 | 资源 | 耗时 | 目的 |
|---|---|---|---|
| A. tiny 打包 + 起服务 | 1 卡 | ~3 min | **决定性验证**：vLLM 是否在 gfx90a 选中 `TritonW4A16LinearKernel`（linear）与 `WNA16MoEBackend.TRITON`（MoE）；若仍落到 emulation/marlin 则此方向作废（代价极小） |
| B. 全量打包 | 1 卡 + 读 806GB / 写 ~200GB | ~15–20 min | 产出 `Ornith-1.5-397B-CT-Int4-W4A16` |
| C. TP8 加载 + 实测 | 8 卡 | ~10 min | 吞吐 / NLL / 显存，与 mxfp4(7.16 tok/s) 和 int8(37.57 tok/s) 对比 |

**预期（需实测确认）**：CDNA2 同样没有 int4 硬件，Triton W4A16 也是"反量化 + bf16 数学"，
但它的反量化**融合在 GEMM 内部**，不像 MXFP4 emulation 那样先把权重物化成 bf16 再算 →
少了 4 倍的中间读写。合理预期比 7.16 tok/s 快 2–4 倍；能否超过 int8 的 37.57 tok/s **未知**。

**可选**：`--group-size 32` 质量更好但 scale 体积 4 倍（默认 128）。
