# A（AITER a8w8 调优）—— 已暂停，断点与恢复方法

暂停：2026-09-15 ~14:50 UTC，按指示"暂停调优"。
（注：`tuning/` 目录由容器内 root 创建，故本文件放在其上层。）

## 暂停时的精确位置

进度日志 `tuning/progress.txt` 最后三行：

```
[14:37:22Z] build=223M cc=1 winners=0/40
[14:42:22Z] build=223M cc=1 winners=0/40
[14:47:22Z] build=442M cc=0 winners=0/40
```

**tune 模块的 JIT 编译已完成**（442 MB；`cc=0` 表示 clang 进程已归零），A 停在
「构建完成 → 13,760 项 GPU benchmark 即将开始」这个界面。`winners=0/40` 说明**尚未产出任何调优结果**，
内存里没有值得保留的进度 —— 因此用 `docker stop` 收尾优于保持 paused（同样不丢东西，还能放掉冻结进程的 RAM）。

## 保留的资产

| 资产 | 位置 | 保留 |
|---|---|---|
| **tune 构建缓存（约 40 分钟编译成果）** | 容器 `hyperloom-srv` 可写层：`/root/.aiter/jit/build/module_gemm_a8w8_tune`（442 MB） | ✅ `stop` 不清可写层；**只有 `docker rm` 会丢** |
| 输入形状清单 | `scripts-local/a8w8_untuned_gfx90a.csv`（40 个 2 的幂档位 M × 6 组 N/K） | ✅ |
| 启动脚本 | `scripts-local/run_a8w8_tuner.sh` | ✅ |
| **已到手的 +17.6% / +15.2% 生产修复** | 见下节校验 | ✅ **完全未受影响** |

## 恢复

```bash
docker start hyperloom-srv
docker exec -d hyperloom-srv bash -lc \
  'bash /home/qiba/ROCm.AI/hyperloom/scripts-local/run_a8w8_tuner.sh > /tmp/a8w8-tuner.log 2>&1'
```

JIT 缓存若命中则跳过约 40 分钟重建、直接进 benchmark；哈希不匹配则重算一次构建。
放弃并清理：`docker rm hyperloom-srv`（丢构建缓存，**不影响生产修复**）。

## 暂停期间无污染校验（逐项实测通过）

调优器设计上只写 `-o` 指定的工作区文件，未触碰生产配置：

| 项 | 值 | 判定 |
|---|---|---|
| 生产 `aiter/jit/module_gemm_a8w8.so` | md5 `65a952b63897ad28387df04cc6ede516` | ✅ 仍是**修复版** |
| `csrc/ck_gemm_a8w8/gemm_a8w8.cu` | 5 处 `M <= 64` | ✅ 在位 |
| 生产 `aiter/configs/a8w8_tuned_gemm.csv` | 580 行，md5 `69040f05…` | ✅ **未被写入** |
| `tuning/a8w8_tuned_gfx90a.csv` | 不存在 | ✅ 无半成品 |
| GPU / 僵尸 | 8/8 空闲、0 僵尸 | ✅ 干净

生产路径（宿主侧；容器内为同名只读 bind）：
`/home/qiba/ai/envs/wu1w-int8-028/lib/python3.12/site-packages/aiter/...`

## 恢复后 A 的产出与关键提醒

- benchmark：40 形状 × 344 候选 = 13,760 项（`warmup=5, iters=101`），单卡
- 产物：`tuning/a8w8_tuned_gfx90a.csv`（每形状胜者）＋ `tuning/profile_a8w8_all_candidates.csv`（**全部候选明细**）
- **关键读数在后者**：能直接回答"M=64 下最优 kernel 比当前 256×128 tile 快多少"以及"splitK 是否真有用"——
  如果最优候选相对现值没有明显优势，就不必再投入重建，A 到此即可判定收益有限。
- **调优结果本身不会自动生效**：查找表 `GENERATE_LOOKUP_TABLE` 是**编译期**从 tuned CSV 生成的，
  必须再用胜者重建推理模块（约 9 分钟）、然后端到端复测，才会变成吞吐。
