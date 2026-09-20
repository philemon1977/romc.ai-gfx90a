# GLM-5.3 CT-INT4：decode 一步的 GPU 时间归属（worker 内 profiler）

**日期**：2026-09-21 02:30 ｜ **配置**：DCP=8 / TP8 / int4 / MAX_MODEL_LEN=32768 / CG=8 / MBT=2048
**被测点**：ctx=8192、M=1（单流 decode），窗口 = 第 25..32 次 execute_model（**纯 decode**）

## 结论（先行）

以 8 个 TP rank 一致的 185–190 ms/步 为分母，decode 的 GPU 时间分布是：

| kernel | ms/步 | 占比 | 调用/步 | 单次 |
|---|---|---|---|---|
| `_sparse_attn_prefill_ragged_kernel`（DSA 稀疏注意力） | **67.2** | **36.0%** | 78（每层 1 次） | 861 µs |
| `ncclDevKernel_Generic_4`（TP all-reduce + DCP 通信） | **36.3** | **19.4%** | 257 | 141 µs |
| `triton_w4a16_gemm_kernel`（**非专家**的 int4 GEMM） | **35.3** | **18.9%** | 261 | 135 µs |
| hipBLASLt `Cijk_...MT64x16x16` | 11.7 | 6.2% | 75 | 155 µs |
| **MoE 专家 GEMV（我们的 v3）** | **3.3** | **1.8%** | 75 | 45 µs |
| MoE 路由/topk/align 等杂项合计 | 11.0 | 5.9% | 642 | |

⇒ **MoE decode GEMV 这条线已经吃完**（1.8%，见 `moe-gemv-scale-hoist.md`）。下一个量级的机会在
**稀疏注意力内核**、**每层集合通信**、**非专家 W4A16 GEMM** 三处。

⇒ **单流 TPS 与上下文强相关**：ctx≈800 时 10 tok/s（≈98 ms/步），ctx=8192 时 ≈4 tok/s（≈250 ms/步）——
报告单流 TPS 必须写上下文长度，否则数字没有意义。

## 方法：三条路只有一条通

| 手段 | 结果 |
|---|---|
| `rocprofv3 --kernel-trace`（图模式 / eager 各一次） | ❌ 两次都在**装载结束、引擎初始化**处死锁：GPU 0% 占用、worker 232% CPU 空转、连 rocprofv3 自己的 signal handler 都挂住 ⇒ 疑与 RCCL/多进程初始化冲突 |
| driver 进程内 `torch.profiler` | ❌ 只有 8.8 µs 的 `hipDeviceSynchronize` —— vLLM v1 把模型跑在**独立 worker 进程**里 |
| **worker 内注入**（本次采用） | ✅ 见下 |

实现：`quark-int8/moe_gemv_patch/sitecustomize.py` 里新增 env 门控钩子（沿用文件里既有的
`MetaPathFinder` 写法），在 worker 进程内挂钩 `GPUModelRunner.execute_model`：

    MI250_PROF_WORKER=1 MI250_PROF_SKIP=25 MI250_PROF_STEPS=8 MI250_PROF_OUT=/work/prof
    ⇒ 抓第 25..32 步；各 rank 写 worker_rank<N>.txt / .json

`quark-int8/scripts_local/glm_prof.sh` 负责起一次性容器（含 8 卡显存硬门与等待释放），
`quark-int8/analyze_worker_prof.py` 做聚合（按类目归并 + ms/步 + 调用次数）。
默认全部关闭，对生产零影响。

## ⚠️ 更正留痕：我先说错了一次

第一次抓的 8 步窗口里**混进了 prefill 分块**（MBT=2048 切 8192 的 prompt ⇒ 4 个分块步），
于是把该内核占 52.7% 读成「decode 在跑 prefill 形状的内核」，并怀疑是 DCP 的行数 bug。
**这个推论是错的**，证据链：

1. 混窗口里有 `vllm::unified_mla_attention_with_output`（635 ms），而**纯 decode 窗口里它根本不出现**
   ⇒ 它属于 prefill；
2. 换 skip=25 的纯 decode 窗口后，该内核仍是 **78 次/步**（每层一次），但这是 DCP 下 DSA 的
   **decode 稀疏注意力内核本身**（名字里的 "prefill" 是历史命名，不代表它在做 prefill 规模的活）；
3. ctx=512 的插桩跑里它整个 run 只被调 6 次 ⇒ 疑似**只有上下文超过 index_topk(2048) 才走稀疏路径**，
   短上下文走 dense。**这条阈值假设尚未专门验证，不要当结论引用。**

## 复现

    # 纯 decode 窗口的 profile
    cd /home/qiba/ROCm.AI && MI250_PROF_WORKER=1 MI250_PROF_SKIP=25 MI250_PROF_STEPS=8 \
      MI250_PROF_OUT=/work/prof CTX=8192 GEN=32 bash quark-int8/scripts_local/glm_prof.sh
    python3 quark-int8/analyze_worker_prof.py quark-int8/prof 24

（注意：prof 目录由容器以 root 创建，宿主机删不掉；换目录或先用一次性容器清理。）

## 顺带发现：仓库里的 .patch 比线上树旧

把 `dcp_patches/base/ops.py.orig` 依次打上 0001/0005 后与线上树 diff，**只有一处差异**：
树上是模块级 `_DCP_TOPK_CTX`（由后端 `__init__` 用 `set_dcp_topk_ctx` 写入），补丁生成的是在 op 里调
`get_current_vllm_config()` 的旧版——那是本会话早先修 `AssertionError: Current vLLM config is not set`
时**只改树、没回写生成器**留下的。⇒ 克隆仓库按 README 打补丁会得到会报错的旧版；**当前以线上树为准**，
收尾时要把该 hunk 写回 `make_patch*.py` 或补一个 0007。
