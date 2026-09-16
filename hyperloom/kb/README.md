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
