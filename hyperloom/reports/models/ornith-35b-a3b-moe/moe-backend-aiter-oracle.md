# `Ornith-1.5-35B-A3B`（MoE，未量化 bf16）— **MoE 在 gfx90a 上是活的**

- 模型：`/mnt/stripe-3mix-3t2/models/ornith-ai/Ornith-1.5-35B-A3B`
- 架构：`Qwen3_5MoeForConditionalGeneration`，**MoE**，A3B（35B 总参 / 3B 激活）
- 权重：**68 GB** bf16，16 分片，`quantization=None`（未量化）
- 结构：混合 **GDN 线性注意力**（`qwen_gdn_linear_attn`）＋ MoE 路由；config 带 `vision_config`，本仓库跑纯文本路径
- 机器：8× MI250X (gfx90a/CDNA2)，ROCm 7.2.4，vllm 0.28.0+rocm723

## 结论

**MoE 路径在 gfx90a 上没有死。** 该模型的失败与 gfx90a 的指令集无关，是 **vLLM 的 MoE 后端选择缺陷**。一个环境变量即可跑通，无需任何 kernel 移植、无需任何反量化。

## 失败原文（`session/Ornith-1.5-35B-A3B/20260914T094955Z-aa171c76/runs/specialist/78b25…/scratch/server.log`）

```
ERROR [multiproc_executor.py:941]  self.experts = FusedMoEFactory(
  fused_moe/layer.py:375                FusedMoEFactory
  fused_moe/routed_experts.py:121       __init__
  fused_moe/routed_experts.py:203       _get_quant_method
    quant_method = UnquantizedFusedMoEMethod(moe_config)
  fused_moe/unquantized_fused_moe_method.py:47
    self.unquantized_backend, self.experts_cls = select_unquantized_moe_backend(
  fused_moe/oracle/unquantized.py:311
    raise ValueError(_make_log_unsupported(backend, reason))

ValueError: Unquantized MoE backend ROCm AITER does not support the deployment configuration
since kernel does not support current device rocm. AITER MoE is not enabled —
set VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_MOE=1 to enable it.

→ RuntimeError: Engine core initialization failed.
```

该 session 因此收在 `enablement_stalled`、`baseline 0.0`（5 次 `RuntimeError: Engine core initialization failed` 告警）。

## 根因：oracle 分支只查 env、不查设备

`vllm/model_executor/layers/fused_moe/oracle/unquantized.py`，ROCm 的候选优先级表（第 61-66 行）把 AITER 排第一：

```python
if current_platform.is_rocm():
    _AVAILABLE_BACKENDS = [UnquantizedMoeBackend.AITER,
                           UnquantizedMoeBackend.TRITON,
                           UnquantizedMoeBackend.BATCHED_TRITON]
```

而第 300-311 行的选择逻辑：

```python
if envs.is_set("VLLM_ROCM_USE_AITER") or envs.is_set("VLLM_ROCM_USE_AITER_MOE"):
    skip_aiter_moe = (not envs.VLLM_ROCM_USE_AITER
                      or not envs.VLLM_ROCM_USE_AITER_MOE
                      or rocm_aiter_ops.is_rdna_aiter_enabled())
    if skip_aiter_moe:
        AVAILABLE_BACKENDS.remove(UnquantizedMoeBackend.AITER)   # ← 会回退
    else:
        backend = UnquantizedMoeBackend.AITER
        return _return_or_raise(backend, moe_config, activation_format)   # ← 直接抛，不回退
```

两处叠加造成必然失败：

1. `VLLM_ROCM_USE_AITER_MOE` **默认 `"True"`**（`vllm/envs.py:1262-1264`：`os.getenv("VLLM_ROCM_USE_AITER_MOE", "True")`）。而 Hyperloom 的 launcher 传了 `VLLM_ROCM_USE_AITER=1` → `skip_aiter_moe=False` → **强行选中 AITER**。
2. 该分支**只看 env，不看设备是否支持 AITER**。而 `is_aiter_found_and_supported()` 要求 CDNA3+（`vllm/_aiter_ops.py:137`：`device arch is CDNA 3 or better`）——gfx90a 是 CDNA2，AITER 不可用。

于是 `_return_or_raise` 抛出那句**自相矛盾**的报错：它叫用户设 `VLLM_ROCM_USE_AITER=1` 和 `VLLM_ROCM_USE_AITER_MOE=1`，而这两个 env **当时都已经设了**。真正缺的是设备支持。

**这个报错在 gfx90a 上是有害的建议**：照它去做，会打开一个 kernel 根本不可移植的 ASM 路径（见 `shared/` 第 5 节）。

## 修复：`VLLM_ROCM_USE_AITER_MOE=0`

显式设 0 后 `envs.is_set(...)` 为真 → `skip_aiter_moe = not False = True` → AITER 从候选移除 → 第 313 行循环落到 **TRITON**。

### 验证（2026-09-15 04:12–04:19 UTC，本机）

唯一自变量是 `VLLM_ROCM_USE_AITER_MOE=0`，其余与失败配置一致：

```bash
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=0,1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=0          # ← 唯一改动
/opt/envs/vllm/bin/vllm serve /mnt/stripe-3mix-3t2/models/ornith-ai/Ornith-1.5-35B-A3B \
  --tensor-parallel-size 2 --port 8899 --trust-remote-code --language-model-only
```

| 检查 | 结果 |
|---|---|
| 越过 `select_unquantized_moe_backend` | ✅ 无 `ValueError`，直接进入权重加载 |
| 后端选择日志 | `[unquantized.py:319] Using TRITON Unquantized MoE backend out of potential backends: ['TRITON', 'BATCHED_TRITON'].` |
| 权重加载 | ✅ 16/16 分片（68 GB，TP=2） |
| server 就绪 | ✅ `GET /health` → **200** |
| 实际生成 | ✅ `finish_reason=stop`，232 completion tokens |
| 粗测单流吞吐 | 232 tok / 2.83 s ≈ **82 tok/s**（单流、warm、非规范基准） |

附带观察：GDN 线性注意力同样走 Triton 回退（`Falling back to the Triton GDN decode path: torch.ops._C.fused_gdn_decode_post_conv_mtp is not built` → `GDN decode kernel: triton`），CK/cuteDSL 路径也不可用。这印证了同一模式：**gfx90a 上真正可用的是 Triton/CK 这类源码编译路径，而不是预编译的 gfx942 ASM 码对象。**

## 配置注意（供后续真基准使用）

本次未传 `--max-model-len`，vLLM 取了模型的 `max_seq_len=262144`，后果是：

```
Available KV cache memory: 9.2 GiB
GPU KV cache size: 938,578 tokens, Maximum concurrency for 262,144 tokens per request: 3.58x
```

即 256k 上下文下最大并发仅 3.58x。要和 dense 模型的 `MAX_MODEL_LEN=6144` 可比，需显式传 `--max-model-len`。上面 82 tok/s 的数字因此只是"能跑"的证明，不是可比的基准数。

## 本条推翻了什么

早期我写过"gfx90a 上 MoE 路径死（缺 bf16 原子）"。**该结论错误，来源是把两件事混为一谈**：

- AITER 的 **ASM kernel 二进制可移植性普查**（1422 个 gfx942 kernel，242 可二进制补丁、539 因缺 `global_atomic_pk_add_bf16` 被挡）——这是**预编译码对象**层面的上限；
- 而该模型的实际运行时失败，发生在**后端选择**这一层，与原子指令、与 FP8/FP16 都无关。

同一逻辑的现成反证：dense INT8 模型上 AITER 那 482 个 int8 ASM kernel 同样被判"不可移植"，它却跑到了 444.9 tok/s。

## 关于"把 FP8 反量化为 FP16 跑通 MoE"这个思路

**打错了层，且前提在本模型上不存在。**

- 本模型是**未量化 bf16**（`quantization=None`），**没有 FP8 可反量化**。
- 原子阻塞的是**累加器**类型，反量化改的是**权重**类型，两者不相干——换成 FP16 权重不会移除任何 atomic 需求。
- 权重字节数 FP8 1B / INT8 1B / **FP16 2B**；decode 是访存受限，权重翻倍直接掉吞吐。gfx90a 上正确的量化是 **INT8 W8A8**（dense 模型已验证的那条路），而非 FP16。
- 该思路**真正**适用的场景是"把只有 FP8 发布版的 checkpoint 搬到 gfx90a 上跑起来"；而在那个场景里 FPGA→INT8 W8A8 仍优于 FPGA→FP16。

**有一半是对的**：`port-matrix.md` 明确记载 gfx90a **有** `global_atomic_pk_add_f16` 和 `global_atomic_add_f32`，只是没有 bf16 打包形式。所以 bf16→fp16 的**累加器重定向**确实是 gfx90a 上一条真实的源码级移植路线——但它的边界要看清：单靠它只解锁 `bf16gemm`（22 个，且仅因 bf16_atomic 被挡）；`fmoe`（838 个）是 **fp8 + int8 + bf16_atomic 三者同时**被挡，`fmoe_2stages`（186 个）是 fp8 + int8。

## 上游可报项

1. **oracle 分支缺设备支持检查**（`oracle/unquantized.py:300-311`）：`VLLM_ROCM_USE_AITER_MOE` 默认 `True` + `VLLM_ROCM_USE_AITER=1` 会在不支持 AITER 的 ROCm 设备上强行选中 AITER 并硬失败，而不是滑到 TRITON。判据里应并入 `is_aiter_found_and_supported()`。
2. **报错文案误导**：错误信息建议开启那两个 env，而在设备不支持时照做会走入更坏的失败。

（此二项属 vLLM 上游，不在 AMD-AGI/Hyperloom。）
