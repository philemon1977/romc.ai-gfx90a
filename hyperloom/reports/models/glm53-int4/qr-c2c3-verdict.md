# QuickReduce C2+C3 在 GLM-5.3-CT-Int4-W4A16 / MI250X 上的三臂实证（2026-09-21）

**结论先行：QR 在本模型上不可用 —— 不是「测不出收益」，而是「开了就起不来」。**
`init_custom_qr` 在 vLLM 做显存规划**之前**就固定吃掉 ~9 GiB/卡，与 `GPU_MEM_UTIL=0.97` 的门
直接冲突，8 个 worker 全部拒启。要让 QR 上场必须把 util 降到 ≤0.858，KV 池只剩 ~2 GiB ⇒ 对
「1M 上下文」这个目标等于自断一臂。**默认保持关闭**（三条 env 留空 = 不注入，见 launcher）。

## 一、三臂设计与结果（同一把尺子：ISL≈800 / OSL 128 / temp0 / 请求级流式，token 数取 usage）

| 臂 | 镜像 | QR env | 事实召回 | conc=1 | conc=8 | conc=32 | 后端选择 |
|---|---|---|---|---|---|---|---|
| A | `...-0918`（stock） | 不设 | 6/6 | 4.38 | 28.15 | 74.77 | `['PYNCCL']` |
| A2 | `...-0918-qr`（含 C2+C3） | 不设 | 6/6 | 4.18 | 26.64 | 73.13 | `['PYNCCL']` |
| B | `...-0918-qr` | 三条全给（FP / 0 / 0） | — | — | — | — | 起不来 |

- **A2 的意义**：证明「补丁代码在、但 env 不设」时不会走 QR 分支（日志里没有任何 quick allreduce
  行，仍是 `['PYNCCL']`）。A2 对 A 的 −2…−5% 落在本机 cross-boot 漂移（记录 ±15%）内 ⇒ 单次对拍
  不足以声称中性，只能说没有反向证据。
- **B 的意义**：证伪「QR 只要装了就能用」。日志证据（`/home/qiba/ai/logs/glm53/server-8121-20260921-130453.log`）：

```
  (Worker pid=781..788) INFO [quick_all_reduce.py:249] Custom quick allreduce: min size override = 0 MB
  (Worker pid=786) ERROR [multiproc_executor.py:943] WorkerProc failed to start.
  ... File "vllm/v1/worker/utils.py", line 542, in request_memory
  ValueError: Free memory on device cuda:5 (54.9/63.98 GiB) on startup is less than
              desired GPU memory utilization (0.97, 62.06 GiB).
```

即：三条 env 确实生效（`min size override = 0 MB` 就是 C3 那条直通的回声），但 init 期的 QR 缓冲
把每卡可用显存压到 **54.9 GiB**，低于 0.97 档要求的 **62.06 GiB**，引擎直接拒绝启动。

## 二、数字账（为什么不值得为 QR 降 util）

- 单卡总量 63.98 GiB；0.97 档要 62.06 GiB；QR 开时启动瞬间只剩 54.9 GiB ⇒ QR 自身约 **8–9 GiB/卡**。
- 要过 `request_memory` 需 `util × 63.98 ≤ 54.9` ⇒ **util ≤ 0.858**。
- 该 util 下权重 52.9 GiB/rank + QR ~9 GiB ⇒ KV 只剩 **≈2 GiB/卡**（32k 档原本 8.17 GiB）。
  按 DCP=8 的 11.5 KiB/token 换算约 18 万 token ⇒ **1M 上下文彻底不可能**。
- 对比收益：QR 换的是 all-reduce 路径（decode 一步 NCCL 占 **19.4%**、141 µs/次），代价是 KV 池与
  全部长上下文能力 ⇒ 在本机目标函数（TPS + 1M）下不划算。

## 三、复现命令（无人值守 A/B，一臂一杠杆）

```bash
SKIP_A2=1 bash quark-int8/qr_ab_watch.sh                 # A 臂：stock 镜像、env 不设
SKIP_A=1  bash quark-int8/qr_ab_watch.sh                 # A2 臂 + B 臂
SKIP_A=1 SKIP_A2=1 bash quark-int8/qr_ab_watch.sh        # 只补 B 臂
```

脚本自带 `wait_free`（八张卡全空且无别的 api_server 才动手）与 fail-fast，每臂日志按臂名+时间戳落盘。
三条 env：`VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB=0` / `..._QUANTIZATION=FP` / `..._CAST_BF16_TO_FP16=0`；
注意**留空 ≠ 不启用**：枚举型 env 注入空串会让 worker 抛 `Invalid value ''`（launcher 已改成非空才注入）。

## 四、什么情况下重开这个结论

1. 上游把 QR 缓冲改成按需/按形状分配（不再固定 ~9 GiB/卡）；
2. 或 `init_custom_qr` 挪到显存规划之后（那它会被计入 KV 预算而不是顶穿门）；
3. 或换模型/形状让权重显著变小，9 GiB 不再是决定项。
重开时用同一把尺子跑 A/A2/B 三臂，并**同时**记录 KV 池 token 数 —— 只看 TPS 会得出错误结论。

## 五、数据出处

- A 臂：`quark-int8/logs/qr_ab_20260921_074831/`（`A_baseline_{recall.txt,tps.jsonl}`）
- A2 臂：`quark-int8/logs/qr_ab_20260921_101907/`
- B 臂：`quark-int8/logs/qr_ab_20260921_130453/` + 完整容器日志
  `/home/qiba/ai/logs/glm53/server-8121-20260921-130453.log`（引用的 ValueError 在其中）
- 镜像 `rocm-ai/vllm:glm53-int4-gfx90a-0918-qr`（`docker commit --change` 重烤）；applier
  `hyperloom/patches-local/apply_gfx90a_quickreduce.py`
- `quark-int8/logs/` 是 gitignore 的运行时目录，本报告引用的数值已抄录在正文。
