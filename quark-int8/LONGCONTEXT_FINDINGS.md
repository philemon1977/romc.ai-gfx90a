# 长上下文（512K/640K）+ 推测解码 结论与实测

机器：8×AMD Instinct MI250X (gfx90a, 64GB/GCD) · 模型：`Ornith-1.5-397B-Quark-Int8-Attn`（Quark 原生 INT8 W8A8，experts + attn/GDN 投影）
运行时：`vllm/vllm-openai-rocm:nightly`（vLLM 0.29.1rc1.dev47）

## 最终固定配置（已写入 `serve_int8.sh`）

**512K 上下文（YaRN 2.0×）+ ngram 推测解码 + 零专家卸载 + CUDA graph**

```
--max-model-len 520000
--hf-overrides '{"max_position_embeddings":524288,"text_config":{... "rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144,"mrope_section":[11,11,10],"mrope_interleaved":true}}}'
--speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_min":5,"prompt_lookup_max":5}'
--max-num-seqs 8 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.975
--language-model-only --trust-remote-code --moe-backend triton
```

## 为什么是 ngram 而不是 MTP（本轮最重要的发现）

vLLM 源码 `v1/core/kv_cache_utils.py:1901`：

```python
def use_eagle(self) -> bool:
    return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")   # ngram 不在其中
```

- **MTP**：`use_eagle()` 为真 → `_warn_if_unannotated_eagle_mamba` 检查"是否有 KV 组被标注为草稿组"；这个混合 GDN/Mamba 模型（Mamba 组 [0,1,2]）**没有任何组被标注** → 兜底把**所有组**当草稿组 → Mamba 组无法满足加宽的查找窗口 → **跨请求前缀缓存静默归零**。
  **实测证据**：`prefix_cache_queries_total = 74,761`，`prefix_cache_hits_total = 0`。
  后果：同一长文再提问 = 全额重算（512K 约 20 分钟、640K 约 35-45 分钟）。此外任何外部 KV 卸载层会"只写不命中"。
- **ngram**：`use_eagle()` 为假 → 该检查直接 return → **前缀缓存保留**。
  **实测证据**：警告计数 = **0**；`prefix_cache_hits_total = 232,288` token；同一 232K 前缀的第二个请求 **173.1s → 17.2s**（ratio 0.100）。
- 附带好处：ngram 不需要草稿模型 → 省回 MTP 权重 1.54 GiB/卡、省掉 "Capturing model for speculator" 步骤；且 512K 只需 2.0× 外推（比 640K 的 2.45× 更安全）。

## 实测对照

| 项 | 32K 基线（MTP, seqs16） | 640K + MTP | **512K + ngram（固定）** |
|---|---|---|---|
| 短上下文解码 | **39.96 tok/s** | 12.26 tok/s | 15.01 tok/s |
| KV 池 | 418,083 | 644,713（seqs 2） | 581,740（seqs 8） |
| 并发（满长请求） | 1.59× @256K | **1.01× @640K** | 1.12× @520K |
| 前缀缓存 | 生效 | **❌ 失效（hits 0）** | **✅ 生效（hits 232,288）** |
| 同一前缀再提问 | — | ≈全额重算 | **173 s → 17 s** |
| RoPE 外推 | 1× | 2.45× | **2.0×** |
| needle @ >原生窗口 | — | 未测 | **✅ 298,978 token 命中** |

needle 明细（512K+ngram 配置）：74.7K ✅ → **298.9K ✅（已超原生 262,144）** → ~499K（测试中）。

## 已知代价与开放问题

1. **预填吞吐随长度急剧退化**：38K 5,586 tok/s → 152K 3,310 → **299K ~615-643 tok/s**（299K 实测 465s 预填）。据此 512K 首次预填约 20 分钟。**这是长上下文的主要成本，前缀缓存是唯一有效缓解手段**（所以必须保留 ngram）。
2. **大 `max-model-len` 让解码从 ~40 → ~15 tok/s**（跨配置的 2.6 倍损失，与 ngram/MTP 无关）。待查：注意力后端选择、aiter paged-attn、KV layout（LBHNC）是否可修回。
3. **640K 的 1.01× 并发意味着输出预算只剩 ~4.7K token** → 实际使用应设 600-620K。
4. **ngram 接受率依赖负载**：抽取/引用/结构化输出命中高，自由创作低。

## 备选配置

`./serve_int8_mtp640k.sh` —— 640K + MTP，**仅适用于"每次都是全新文档、单发"**的场景（前缀缓存失效无所谓，且 MTP 解码在短上下文下更快）。注意把 `MAX_LEN` 设到 600-620K 以免输出空间被吃光。

## 已被否决的路线（附证据）

| 路线 | 结论 |
|---|---|
| host 驱动的专家槽位缓存（`expert_cache/slot_cache_v2.py`） | **功能成功**：8% 卸载释放 3.15 GB/卡，KV 池 418,083 → **755,347 tokens**（+80%），640K 显存可行。**但**槽位决策是 host 侧数据依赖 → 必须 `--enforce-eager` → 实测 eager 本身就让 batch=1 解码从 39.96 → **9.76 tok/s**（4 倍损失），缓存再加 ~20%。对低延迟长上下文不可行；仅在大批量吞吐（sync 开销被摊薄）时才值得。 |
| 深度卸载（>50% 专家） | 路由分布很平（最热 5% 实例只覆盖 8.2% 专家），且相邻 token 重合度仅 0.294 → PCIe 流量按比例爆炸，不可掩盖。 |

## 相关文件

- `serve_int8.sh`（固定配置） / `serve_int8_mtp640k.sh`（备选）
- `ROUTING_REPORT.md` + `routing_samples.npz`：路由热度/命中率曲线（300 万专家实例）
- `prefix_reuse_probe.py`：前缀缓存复用验证
- `longctx_test.py`：needle 检索测试
- `nll_probe.py`：质量（平均 token NLL）
- `logs/ngram_512k.log`、`logs/speed_iso.log`、`logs/c_needle.log`：原始实验日志
- `EVIDENCE_*.txt`、`NLL_*.json`：各版本推理/吞吐/质量证据
