# `aiter-cpp_extension-pybind-abi.patch` 的来源与定位（2026-09-21 入库）

**来源**：原本只存在于 `/home/qiba/ai/recipes/patches/vllm/vllm_0.28.0_rocm72/aiter-cpp_extension-pybind-abi.patch`（补丁树 README 记为
"手工修法"），2026-09-21 按"负结论也要入库"的纪律拷入本仓。

**它解决什么**：AITER JIT 的 pybind11 internals 在 v11/v12 两版之间的分裂（改 `cpp_extension` 的 pybind 接口假设）。

**结论：是负结果证据，不是可用修复。** 补丁树的原文定位是"AITER MoE 终案 ❌，留作证据" ——
即 AITER 的 MoE 路径在 gfx90a 上不可用这一结论的证据链之一；本仓的现役配置一律
`VLLM_ROCM_USE_AITER_MOE=0`（见 launcher）。**默认不应用**，仅为可复现保留。

**与本仓其它 AITER 资产的关系**：
- 可用且已入库：`hyperloom/reports/models/qwen38-27b-w8a8-dense/a8w8-fix/gemm_a8w8_M64_boundary.patch`
  （CK gemm_a8w8 的 `M<64` off-by-one，实测 +17.6%）+ 两个 `.so`（走 LFS）；
- 不可用（证据）：本文件。
