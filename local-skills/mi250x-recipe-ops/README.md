# mi250x-recipe-ops（本地 AMD Skill）

本机 8×MI250X 工作站的配方操作技能。结构与 `amd-skills/skills/*`（官方
[amd/skills](https://github.com/amd/skills)）一致：`SKILL.md` front-matter +
`skill-card.md` + `data/*.json`。

- 权威来源：`/home/qiba/ai/docs/recipes/`（49 条，闸口 `tools/audit_recipes.py`）。
- `data/*.json` 是机器可读摘录；冲突时以配方 markdown 与 launcher 实物为准。
- 重新生成：`python3 /home/qiba/ROCm.AI/scripts/build_skill_data.py`
  （需先刷新 `.tmp/kb_extract/` 的抽取）。

## 安装（可选）

```bash
ln -s /home/qiba/ROCm.AI/local-skills/mi250x-recipe-ops ~/.dsh/skills/mi250x-recipe-ops
```

## 配套资产

- Hyperloom RecipeKB 语料：`/home/qiba/ROCm.AI/hyperloom/kb/`（见 `data/recipe_kb.json`
  与 `kb/seed_manifest.json`）。
- 播种/校验：`scripts/seed_recipe_kb.py`、`scripts/verify_recipe_kb.py`。
