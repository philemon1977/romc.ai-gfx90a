# 已按模型拆分（本文件不再维护）

这份报告写作时把两个模型的结论混在了一起，已拆分为：

- dense INT8 模型 → [`models/qwen38-27b-w8a8-dense/baseline-validated-20260914.md`](models/qwen38-27b-w8a8-dense/baseline-validated-20260914.md)
- MoE 模型 → [`models/ornith-35b-a3b-moe/moe-backend-aiter-oracle.md`](models/ornith-35b-a3b-moe/moe-backend-aiter-oracle.md)
- 与模型无关的部分（编排缺陷、预算算术、环境/机架观测）→ [`shared/orchestration-budget-and-teardown.md`](shared/orchestration-budget-and-teardown.md)

总索引见 [`README.md`](README.md)。

拆分理由：这两个模型的失败**形态相同、根因不同**（dense 死在掩码泄漏，MoE 死在 vLLM 的 MoE 后端选择）。混在一起讲会得出"gfx90a 上 MoE 路径死"这种已被推翻的结论。
