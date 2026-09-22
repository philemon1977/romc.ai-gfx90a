# Ornith-1.5-397B INT8-Attn：投机解码 × 并行拓扑 —— 实测报告（2026-09-22）

> 本报告是"结论 + 数字"的家；**死路表只留判语与指针**（`local-skills/mi250x-recipe-ops/references/50-dead-routes.md` 同日条目）。
> 原始产物（日志、逐步输出、编排脚本）在 `$AI/bench/ornith-tp2pp4-ab-20260922/` 与 `$AI/bench/ornith-dflash-vs-mtp-20260922/`。
> 服务臂配方：`$AI/docs/recipes/serving/8127-ornith-1-5-397b-int8-{dflash15-vllm-tp8,vllm-tp4pp2-spec0,vllm-tp2pp4-spec0}.md`。

## 0. 一句话

在 8×MI250X 上，**Ornith-1.5-397B INT8-Attn（TP8 + 内置 MTP(5)）仍是最优服务形态**；
用户提出的"TP2×PP4 会不会更快"**实测更慢（单流 −17.6%）**；
社区 DFlash 草稿**按负载分裂**（可预测负载 +72.7%、散文 −27.4%）；
过程中反解出两条可复用机理：**步时 90% 是与并行宽度无关的延迟**、**KV 容量 ∝ PP 级数**。

## 1. 口径（全部同一次 boot，背靠背）

- 机器：8×MI250X（4 模组 × 2 die，gfx90a），ROCm 7.2.4，vLLM 0.28.0+rocm723，`envs/vllm_0.28.0_rocm72`
- 模型：`models/ornith-ai/Ornith-1.5-397B-Quark-Int8-Attn`，**385.71 GiB / 123 片**
  （逐片头解析：**专家 372.70 GiB = 96.6%**、attn/GDN 8.30、embed/lm_head/norm 4.70）
- 测量：`measure_median.py` 固定 prompt + **丢弃首请求** + 中位数 n=5 / 256 token（端到端含 prefill）；
  并发档 `bench_concurrency.py` 400-token prompt / 128 out。**跨 boot 绝对值不可混表**。
- 端口 8127（实验位），每次起服前 8 die 全空（脚本自带 fail-closed 门）。

## 2. 拓扑曲线（SPEC=0，同 boot 三点）

| 拓扑 | 落位（实测 worker 名） | 单流 count | step_ms | 并发 4 | 并发 8 | KV @262144 |
|---|---|---|---|---|---|---|
| **TP8·PP1** | `Worker_TP0..7` | **42.86 t/s** | **23.33** | 102.24 | **203.94** | 748,908 |
| TP4×PP2 | `Worker_PP{0,1}_TP{0..3}` | 41.18 | 24.28（−3.9%） | **104.12** | 147.51（−27.7%） | 1,478,286 |
| TP2×PP4 | `Worker_PP{0..3}_TP{0,1}` | 35.33 | 28.30（−17.6%） | 60.75 | 104.02（−49.0%） | 2,712,793 |

- **落位如实成立**：vLLM 约定 `rank = pp*TP + tp`、`local_rank = rank = die号`
  ⇒ TP2 组 = 四个 OAM 模组（模组内 144 GB/s xGMI）；TP4 组 = 一个 NUMA node（组内无 16/SYS 腿）；**无需任何可见性变量**。
- **TP8 最优**；TP4×PP2 的价值在"≤4 并发等价 + KV ×2"。

## 3. 投机解码：MTP(5) vs DFlash(n=15)（TP8，同 boot 背靠背）

| 指标 | MTP(5) | DFlash(15) | Δ |
|---|---|---|---|
| 单流 count | 97.28 t/s | **168.03** | **+72.7%** |
| 单流 explain | **76.10** | 55.26 | **−27.4%** |
| 接受长度 count / explain | 3.9 / 3.0 | **9.88** / 3.0 | +2.6× / 持平 |
| 草稿吞吐 explain | 125 tok/s | **276** | 白付 |
| KV @262144 | 352,826 | **640,086** | +81% |
| Model Runner | V1 | **V2**（强制） | — |

- 草稿 `z-lab/Qwen3.5-397B-A17B-DFlash`（2.41 GiB）base = `Qwen/Qwen3.5-397B-A17B`，
  与本机 Ornith **逐项同构**（arch/vocab 248320/hidden 4096/60 层/512 专家 top-10/MTP 头 1）；
  draft = `DFlashDraftModel`（6 层、block_size=16、8 个 target_layer_ids、无 embed/lm_head）。
- **强制 V2 的原因**：草稿 `layer_types` = 5×sliding + 1×full ⇒ `_dflash_needs_multi_kv_group()`（`config/vllm.py:636`）。
- ✅ 附带资产：**AITER int8 线性补丁在 V2 runner 下判据仍全绿**（后端级、与 runner 无关）。

## 4. 两条可复用机理（本次最有价值的部分）

### 4.1 步时成分：90% 是与并行宽度无关的延迟
用 A（TP8：a+L=23.33 ms）与 B（TP2×PP4：4a+L'≈28.30 ms）联立 ⇒ **a≈2.3 ms（~10%）/ L≈21 ms（~90%）**。
再用交付态（38.7 ms/步、4.20 tok/步）与 A 联立 ⇒ **F≈18.5 ms/步、m≈4.8 ms/token**（跨 boot，仅量级参考）。
⇒ **本臂的杠杆是"减少每步核数/启动次数"与"每步多出 token"，不是并行拓扑**。

### 4.2 KV 容量 ∝ PP 级数
| 拓扑 | 预测 = 748,908 × PP级数 | 实测 | 偏差 |
|---|---|---|---|
| TP8·PP1 | 748,908 | 748,908 | +0.0% |
| TP4×PP2 | 1,497,816 | 1,478,286 | −1.3% |
| TP2×PP4 | 2,995,632 | 2,712,793 | −9.4%（≈其自身 `padding layers 9.09%`） |

机理：**TP 不切层** ⇒ TP8 时每张 die 要为全部 15 个注意力层备 KV；PP 后每 rank 只管 `15/PP级数` 层。

## 5. 内存门（决定"哪些组合根本不用试"）

每 die 权重预算 ≈ **48.2 GiB**（实测：util 0.96 的 61.43 − graph 3.45 − KV 5.4）⇒
**专家 372.70 GiB 必须切 ≥8 份**，且**稠密部分 13.0 GiB 复制不起**：

| 组合 | 每 die 权重 | 判定 |
|---|---|---|
| TP8 / TP4×PP2 / TP2×PP4 | 48.2 / 49.9 / 53.1 | ✅ |
| TP4+DP2+EP8 | 49.9（KV 4.2 GiB/副本 ×2） | ⚠️ 唯一可行的 EP 方案 |
| TP2+DP4+EP8 | 53.1（KV ≈1 GiB） | ❌ 262k 一条请求都服务不了 |
| **TP1+DP8+EP8** | 59.6（稠密全复制） | ❌ 63.4 + act/graph ≈ 70.7 > 63.98 |
| TP2+DP2+EP4 | 99.7（专家只切 4 份） | ❌❌ |
| DP4×TP2（每模组一个副本） | 192.9 | ❌ 需整模型 ≤ ~96 GiB 才成立 |

## 6. 下一步（未做，按用户口径待批）

1. **DFlash n 扫描（3/7/15）× count/explain**：靶子 = explain 上不亏（MTP 76.10 t/s）。
2. **ngram + TP4×PP2**：PP 下唯一能带投机解码的通路，同时验证 KV ×2 是否可吃。
3. 把"按负载选草稿"机制化（client 逐请求指定 n 或按 workload 分臂）。
