# Hyperloom RecipeKB（本机种子语料）

本目录是 Hyperloom 推理优化器的 **本地 Recipe KB 根**（local store 模式），
由 `/home/qiba/ai/docs/recipes/` 的在役/实验服务臂配方种子化而来。

- 布局（local_store 契约，勿手改）：
  `<model>/<hardware>/<framework_name>/<model_type>/<architectures>/<framework_version>/<precision>/recipe.json`
  （+ `history/`、`attempts.ndjson`、`.lock`）
- 硬件维钉死 `mi250x`（单机 8×MI250X，`kb_hardware_slug` 单机直通）。
- framework_version 用本机实物版本串（dist-info / `llama-server --version`）。
- 消费方式（跑 Hyperloom 前）：

  ```bash
  export KNOWLEDGE_LOCAL_ROOT=/home/qiba/ROCm.AI/hyperloom/kb
  ```

- 再生成 / 校验：

  ```bash
  python3 /home/qiba/ROCm.AI/scripts/seed_recipe_kb.py
  python3 /home/qiba/ROCm.AI/scripts/verify_recipe_kb.py
  ```

- 同一 7 元组的多臂（如 8107-vLLM/8109-splitKV、8301/8302）合并为一行：
  `provenance.merged_from` 记来源，`best_config`/`best_throughput` 取实测更优方。
- `seed_manifest.json` 是本次播种的 cid 清单。
- 注意：`/home/qiba/ROCm.AI/hyperloom/session/knowledge/` 是早前容器会话（root
  属主）写的旧根，与本库无关，勿混用。

## 两个写入者（2026-09-21 补）

本库里的 recipe 行有**三个来源**，别把其中一个当成全部：

| 来源 | 覆盖 | 入口 |
|---|---|---|
| 服务臂播种 | `docs/recipes` 抽出的 12 条（llama.cpp / vLLM 在役臂），`seed_manifest.json` 记的就是这些 | `scripts/seed_recipe_kb.py` |
| Hyperloom 自己在跑 | 优化器 CLOSE 时写的槽位（从 t0_anchor 起） | Hyperloom 本体 |
| 本会话实测回填 | GLM-5.3-CT-Int4-W4A16 / mi250x（Hyperloom 建的槽位原是空壳） | `scripts/note_glm53_int4_kb.py` |

- `verify_recipe_kb.py` 只要求磁盘上的行可读、可 search；**行数可以多于
  `seed_manifest.json`**（实测 18 vs 12），多出来的是后两者写的。
- 两个写 recipe 的脚本都必须用 `LocalRecipeStore.put_recipe`（不要手写 `recipe.json`），
  否则 `history/vN.json` 归档与 `version` 会脱节。
- 手写最易踩的两处 schema 坑（2026-09-21 实际踩过）：`remaining_gaps` 的条目必须是
  dict（`description`/`metrics`），写字符串会被**静默丢弃**；`kernel_optimizations` 是定长
  dataclass（`kernel_id`/`source_file`/`micro_speedup`/`e2e_gain_pct`/`e2e_tput`/`decision`/
  `e2e_decision`/`integrated`/`ts`），键名不对会得到一串全零条目。
- 曾经踩过的权限坑：Hyperloom 容器以 root 写的槽位是 `root:root 0600`，宿主（uid 1000）
  读不到 ⇒ `local_store.search()` 直接抛 `LocalRecipeStoreError`，校验失败。修法：容器内
  以 root `chown -R 1000:1000` + `chmod 644`。

