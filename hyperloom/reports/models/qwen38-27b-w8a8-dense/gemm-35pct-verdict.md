# 追 +35%：把 M=64 的 INT8 GEMM 定位到底，并逐条否决可用杠杆

> **后续更新（同日 17:20–18:10）：本报告的"需要内核工程"结论已被执行，并取得 +17.6% 端到端。**
> 见 `a8w8-M64-boundary-fix.md`。一句话：根因是 AITER `gemm_a8w8.cu` 派发启发式的 **`M < 64` 边界**
> 把 M=64（conc 64 的 decode 批量）排除在"小 M 核"之外，落入为大 M 准备的 256×128 tile；
> 改成 `M <= 64` 并重建模块后，output_throughput 439.48 → **516.61（+17.6%）**，TPOT −16.1%，数值在容差内。
> 本报告中被否决的那些**存量配置杠杆**结论依然成立（它们确实都不通）；它们不通的原因正是
> ——真正的杠杆在编译期派发逻辑里，不在运行时配置面。

日期 2026-09-15 07:20–09:20 UTC。目标：把 `profiling-decode-attribution.md` 指出的
"decode 60% 在 INT8 GEMM、只跑到其地板 57%" 变成实际的 +35%（conc 64 从 474 → 约 640 tok/s）。

**结论：机制已完全定位，但在这套栈上没有任何可用杠杆能拿到它。+35% 需要为 gfx90a 从源码构建 CK a8w8 kernel 变体，属内核工程项目，不是调参/调优能解决的。**

## 一、根因：AITER 的 a8w8 调优库没有 gfx90a 行

`aiter/ops/gemm_op_a8w8.py:441 get_CKGEMM_config()` 按 `(gfx, cu_num, padded_M, N, K)` 查
`configs/a8w8_tuned_gemm.csv`。实测：

| 项 | 值 |
|---|---|
| CSV 行数 | 579 |
| CSV 的 `gfx` 取值 | **只有 `gfx942` / `gfx950`** |
| CSV 的 `cu_num` 取值 | 只有 `80` / `256` |
| 本机 `get_gfx()` | **`gfx90a`** |
| 本机 `get_cu_num()` | **`104`** |

→ 查表**必然落空**。运行时证据：单次 server 运行日志中

```
[aiter] shape is M:64, N:16384, K:5120, q_dtype_w:torch.int8,
        not found tuned config in .../aiter/configs/a8w8_tuned_gemm.csv, will use default config!
```

出现 **454 次**；生产环境 `/opt/envs/wu1w` 与 `/opt/envs/vllm` 两份 CSV 都一样（都只有 gfx942/gfx950）。
真实 decode 形状（从日志取得）：`M:1..64 × N∈{5120,6144,14336,16384,17408,34816,248320}, K∈{5120,6144,17408}`。

## 二、未调优时走什么

`gemm_a8w8_CK`：

```python
ck_config = get_GEMM_config_with_quant_type(m, n, k, q_dtype_w, AITER_CONFIG_GEMM_A8W8_FILE)
if splitK is None:
    if ck_config is not None: splitK = ck_config["splitK"]
    else:                     splitK = 0        # ← 本机情况
```

即：**这条路径上调优库唯一能决定的就是 `splitK`**（`kernelId` 并未传入 `gemm_a8w8_ck`）。

## 三、逐条否决（每条都实测）

### ① `--linear-backend` 扫描 —— 已否决
见 `linear-backend-sweep.md`：`auto`/`aiter`/`torch` 稳态中位数**精确同为 553.5 tok/s**（同一路径），
`triton` 慢 21%（435.0）。

### ② splitK —— 本机不可用
`scripts-local/probe_a8w8_splitk.py`（在**生产环境的预编译模块** `/opt/envs/wu1w/.../module_gemm_a8w8.so`
上运行，无需构建）。M=64、逐形状：

| shape | N | K | splitK=0 | splitK=1/2/4 |
|---|---:|---:|---|---|
| o_proj | 5 120 | 6 144 | 0.14 ms / **222 GB/s** | **RuntimeError** |
| mlp_mid | 14 336 | 5 120 | 0.23 ms / **314 GB/s** | RuntimeError |
| qkv | 16 384 | 5 120 | 0.24 ms / **353 GB/s** | RuntimeError |
| gate_up | 34 816 | 5 120 | 0.37 ms / **483 GB/s** | RuntimeError |
| down | 5 120 | 17 408 | 0.37 ms / **238 GB/s** | RuntimeError |
| lm_head | 248 320 | 5 120 | 2.30 ms / **554 GB/s** | RuntimeError |

**gfx90a 的预编译模块没有 splitK 变体**（gfx942 的 ASM a8w8 码对象叫 `gemm_a8w8_m128_noSplitK.co`
/ `gemm_a8w8_m128_splitK.co`，是 M=128 专用且只存在于 `aiter_meta/hsa/gfx942`；
`aiter_meta/hsa/` 下**只有 gfx1250/gfx942/gfx950，没有 gfx90a**）。

⇒ 调优库能提供的唯一字段在本机不可用，**"补 CSV 行"这条最廉价的路被否决**。

### ③ 带宽随 N 单调上升 ⇒ 是占用率受限，不是访存受限
222 → 238 → 314 → 353 → 483 → 554 GB/s，严格随 N 递增。
`BlockN=128` 时 N=5120 只有 **40 个 CTA**，而本机 **104 个 CU**。对照可达值：探针 A 的 M=1 gemv = 721 GB/s、
纯读 = 1372 GB/s。所以这不是"读不动"，是**网格填不满**。

### ④ 更细的 tile / 指定 kernelId —— 运行时不可达
`gemm_a8w8_ck(XQ, WQ, x_scale, w_scale, Y, bias, splitK)` **不接受 kernelId**；
tile 配置（如 `256x128x128`）烘焙在预编译 .so 里。运行时没有选择接口。

### ⑤ bpreshuffle —— 双重复合否决
- **没有 gfx90a 预编译版本**：运行时报
  `[module_gemm_a8w8_bpreshuffle] prebuilt .so targets ['gfx942','gfx950'] but not the running arch gfx90a; rebuilding for gfx90a`（触发另一轮长 JIT 构建）。
- **即便更快，vLLM 也用不上**：vLLM 里确有 helper
  `_rocm_aiter_preshuffled_per_token_w8a8_gemm_impl`（`_aiter_ops.py:683-697`，内部调 `aiter.gemm_a8w8_bpreshuffle`），
  但它**没有注册成 op**（注册的只有 `rocm_aiter_gemm_a8w8_blockscale` / `..._blockscale_bpreshuffle`，那是 fp8 blockscale 用的），
  且在 `model_executor/` 中**没有任何消费者**——对 compressed-tensors 的 per-token W8A8 而言是死代码。

## 四、判定

| 杠杆 | 状态 |
|---|---|
| 并发/调度 | 已量完：conc 32–128 全平（+4.9%） |
| `--linear-backend` | 已否：三取值同路径，第四更差 |
| AITER 调优库 | 无 gfx90a 行；且只暴露 splitK |
| splitK | 本机内核变体不存在（RuntimeError） |
| tile / kernelId | 运行时无接口 |
| bpreshuffle | 无 gfx90a 构建 **且** vLLM 未接线 |

**所以 +35% 在这套栈上不可通过配置或调优获得。** 要拿到它，唯一路径是**为 gfx90a 从源码构建
CK a8w8 的 kernel 变体**（更细的 BlockN/BlockM 以提高网格占用，和/或 splitK 变体），即：
拿到匹配版本的 AITER 源码树 → 构建 tune 模块 → 调优 → 用胜者重建 inference .so → 端到端复测。

规模与风险（诚实估计）：这需要数次 CK 长构建（本机 JIT 一次 `module_gemm_a8w8` 就跑了 30+ 分钟且未完成），
总投入 **数小时**，并且**成功与否取决于 CK 模板能否为 gfx90a 产出更细网格的可用实例**——这一点尚未证明。
收益上限仍是那 +35%（受"每步必须读一遍 27.9 GB 权重"约束，dense INT8 不可能数倍）。

## 五、附带确认的可用资产（供后续内核工作直接使用）

- 生产环境 `/opt/envs/wu1w` 已具备 gfx90a 版：`module_gemm_a8w8.so`(23.6 MB)、`module_gemm_a8w8_asm.so`、
  `module_gemm_a8w8_bpreshuffle.so`、`..._cktile.so` 等全套（后几者在 gfx90a 上**未预编译**，会触发重建）。
- AITER 调优器入口：`python3 aiter/utility/pretune.py module_gemm_a8w8_tune`
  （框架在 `aiter/utility/pretune.py` + `base_tuner.py:GemmCommonTuner`；需 csrc 源码树，wheel 未含）。
- 微基准脚本已就绪：`scripts-local/probe_a8w8_splitk.py`、`scripts-local/probe_a8w8_bpreshuffle.py`。

## 六、给上游的三条（都不是 Hyperloom）

1. **AITER**：`configs/a8w8_tuned_gemm.csv` 无 gfx90a 行 → gfx90a 上所有 a8w8 GEMM 静默退化为默认 config。
   至少在 `get_CKGEMM_config` 未命中时给出更醒目的告警（当前是 INFO 级、每次形状都刷一行，易被淹没）。
2. **AITER**：`splitK>0` 在 gfx90a 上抛 `RuntimeError` 而非回退到 `splitK=0`；对调用方是硬失败。
3. **vLLM**：`_rocm_aiter_preshuffled_per_token_w8a8_gemm_impl` 是未注册的死代码；
   per-token W8A8 在 ROCm 上因此无法使用 bpreshuffle 快路径。

## 产物

- `scripts-local/probe_a8w8_splitk.py`（splitK 扫描；结果见上表）
- `scripts-local/probe_a8w8_bpreshuffle.py`（bpreshuffle 对比；因 ⑤ 未走完）
- 容器内日志：`/tmp/splitk-wu1w.log`、`/tmp/bpre.log`
- 上游上下文见 `profiling-decode-attribution.md`、`linear-backend-sweep.md`
