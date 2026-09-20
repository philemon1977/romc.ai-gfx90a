# GLM-5.3 CT-INT4：MoE GEMV 的 scale 取用是 4 倍级瓶颈（v3 重写）

**日期**：2026-09-20 晚 ｜ **模型**：GLM-5.3-CT-Int4-W4A16 ｜ **硬件**：8×MI250X (gfx90a)
**框架**：vLLM 0.3.1.dev85+gdee37d891（ROCm nightly-0918 镜像 + 本仓 7 个补丁）

## 结论（先行）

MoE decode GEMV 的低带宽**不是** ALU 受限、也**不是**归约开销，而是 **group scale 的逐元素
gather 下标**让 Triton 无法向量化，把取指数放大 ~16 倍。把 scale 提到 k 循环外（每 group 只取
一次）后：**gemm1 8.7×／gemm2 5.0×（生产分片形状）**，端到端**单流解码 6.4–6.8 → 9.6–10.7 tok/s**、
**并发 32 聚合 33.8 → 59.1 tok/s**，事实召回仍 **6/6**。

| 指标 | 改前（v1 内核） | 改后（v3） |
|---|---|---|
| 单流解码 | 6.38–6.81 tok/s | **9.60–10.71 tok/s** |
| 并发 4 聚合 | 13.82 | **20.43–21.73** |
| 并发 8 聚合 | ~20 | **40.64** |
| 并发 32 聚合 | 33.80 | **58.59–59.08** |
| 生产形状 gemm1 | 391.4 us/层 | **45.2 us/层**（8.7×） |
| 生产形状 gemm2 | 132.9 us/层 | **26.5 us/层**（5.0×） |
| 事实召回 | 6/6 | **6/6**（回答逐字一致） |

## 问题：MoE GEMV 只跑到 HBM 的 3.8%

微基准（真实 checkpoint 布局 [E,N,K/2]、gs=32、单 GCD）测得旧内核 **60.1 GB/s**，
而生产形状下 v1 是 **524 us/层 ⇒ 78 层 = 40.9 ms/token**，占当时 147 ms 单步的 **28%**。

## 定位：三个假设，两个被实测证伪

| 实验（quark-int8/moe_gemv_diag.py） | 结果 | 判定 |
|---|---|---|
| 全量内核（v1/v2） | 1674.5 us / 60.1 GB/s | 基线 |
| **去掉 scale 乘法** | **378.2 us / 266.2 GB/s** | **真因在此（4.4×）** |
| 去掉 nibble 解码 | 1663.9 us / 60.5 GB/s | nibble 几乎免费 |
| 纯 load（同访存模式） | 112–162 us / 620–898 GB/s | 访存模式能到 900 GB/s |
| 二维累加器（v2，去掉循环内跨 lane 归约） | 1.01× | **证伪**「归约是主因」 |
| 算术账 | 0.9 Tops/s = fp32 峰值 4% | **证伪**「ALU 受限」 |
| n_regs / n_spills | 73 / 0 | 无寄存器溢出 |

真因：v1 的 scale 下标是逐元素的 kk // GROUP（kk = k0 + 2c + j）。这个非线性下标使 Triton
退化为 [BLOCK_N, BLOCK_K//2] 次 **2 字节 gather**（BLOCK_K=64 时 2048 次取指、其中只有 2 个
不同地址）。注意：**同一模式在"纯 load"下能跑 900 GB/s**，所以问题不在访存模式本身，而在取指。

## 修法：v3（moe_gemv/mi250_moe_gemv_v3.py，已内联进 mi250_moe_gemv_gs.py）

    BLOCK_K = G_PER_STEP * GROUP
    三维累加器 acc3[BLOCK_N, G_PER_STEP, GROUP//2]   # k 循环内完全不碰 scale
    每步收尾：acc += sum_3(acc3) * sc[BLOCK_N, G_PER_STEP]   # 只取一次 scale 切片

开关：MI250_MOE_GEMV_KERNEL=v3（默认）/ v1；选型见 _v3_cfg()（按 N、K、pairs 自动挑）。

### 对拍（vs fp32 反量化参考，全部 0.14%，与旧内核同级 ⇒ 精度不变）

| 形状 | v1 | v3 最优 | 提速 |
|---|---|---|---|
| 全 N：gemm1 K=6144 N=4096 | 1694.5 us | 412.3 us（BN=64 G=8 w=4） | 4.1× |
| 全 N：gemm2 K=2048 N=6144 | 891.1 us | 208.5 us | 4.3× |
| **生产分片**：gemm1 K=6144 N_local=512 | 391.4 us | **45.2 us**（BN=16 G=8 w=1） | **8.7×** |
| **生产分片**：gemm2 K=2048 N_local=768 | 132.9 us | 26.5 us | 5.0× |

生产形状不是猜的：MI250_MOE_GEMV_DEBUG=1 的 DUMP 行实测确认
（A(8,6144) C(8,8,512) B(256,512,3072) Bs(256,512,192) pairs=64 top_k=8 ⇒ N_local=512、gs=32）。
⇒ 网格只有 8×8=64 个 program（104 CU 严重欠占用），所以**小 BLOCK_N 换更多 program** 比
「每 program 干得多」重要得多——按全 N 调出的 BN=64 在分片形状下慢一倍。

## 复现

    # 定位实验（单 GCD，空闲卡）
    docker run --rm --entrypoint python3 --device=/dev/kfd --device=/dev/dri \
      --security-opt seccomp=unconfined --group-add video --ipc=host --shm-size=16g \
      -e HIP_VISIBLE_DEVICES=0 \
      -v <MODEL>:/models:ro -v <PATCH_ROOT>:/patches:ro -v quark-int8:/work:ro -w /work \
      rocm-ai/vllm:glm53-int4-gfx90a-0918  moe_gemv_diag.py --tokens 1,8

    # 生产形状调优 / 对拍
    ... moe_gemv_prod_bench.py --tokens 1,8 --check
    ... moe_gemv_shape2_bench.py        # gemm2 两种分片假设 + 大 pairs

    # 端到端（硬门 + 图模式 + FST 快装载都在脚本里）
    CONC="1,4,8,16,32" bash quark-int8/scripts_local/glm_conc_sweep.sh
    python3 -u quark-int8/fact_recall_probe.py 8122 glm-5.3

## 负结论（别重走）

- **DCP=0（完全不开 DCP）单流并不更快**：7.66 vs DCP=8 的 9.60 tok/s，且 KV 池从 326,016 掉到
  64,176 ⇒ 没有理由放弃 DCP=8。
- **调优在生产里淹没在噪声里**：MoE GEMV 从 28% 降到 ~5% 后，BN=64→16 在探针上只体现为
  6.92 → 7.00 ⇒ 该探针分不出 <5% 的改动，别拿它当判据。
- **gemm2 是否真被接管尚无证据**：DEBUG 打印落在 takeover % 200 == 1，恒为奇数次调用（=gemm1），
  所以那些 apply_w=False 行不能证明 gemm2 没接管；要按 kind 计数重打或单测第二次调用。
- **rocprofv3 与本配置不兼容**：`rocprofv3 --kernel-trace` 两次都在装载结束、引擎初始化处卡死
  （GPU 0%、worker 232% CPU 空转，signal handler 也挂在里面）；图模式与 eager 模式均如此，
  疑与 RCCL 初始化/拦截冲突。kernel 明细改走「worker 进程内 torch.profiler」或其它路径。

## 遗留与下一步

- 单步 147 ms 里 MoE GEMV 只剩 ~5.5 ms ⇒ 剩下 95% 尚未定位（这是下一个 4 倍级机会所在）。
- MTP 投机解码：模型自带 num_nextn_predict_layers=1、checkpoint 含 layers.78 全部张量、
  vLLM 能解析（Resolved architecture: DeepseekV32MTPModel），但**装载期 OOM**（见同目录 MTP 记录）。
- 已知代价提示：vLLM 的 use_eagle() 对 mtp 返回真，在 DSV4.1（Mamba 混合）上曾导致前缀缓存静默清零；
  本模型是纯稀疏注意力，需用 metrics 实测确认后再启用。

## 路径对照

| 本机路径 | 仓库内对应物 |
|---|---|
| /home/qiba/ai/patches/gfx90a/ct_w4a16_dsv41_n0918/moe_gemv/mi250_moe_gemv_gs.py（线上补丁，挂载进容器） | quark-int8/moe_gemv_patch/mi250_moe_gemv_gs.py |
| 同上 mi250_moe_gemv_v3.py、sitecustomize.py | quark-int8/moe_gemv_patch/ 同名文件 |
| /home/qiba/ai/models/ZhipuAI/launcher/glm53_..._mi250dx8.sh（起服脚本，新增 3 个 env 透传） | 见 quark-int8/DCP_A_NOTES.md 的说明 |
