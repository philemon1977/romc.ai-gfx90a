# -*- coding: utf-8 -*-
"""用 vLLM 自己的 should_ignore_layer 判定 ignore 表是否与量化策略一致（与 ① 补丁同调用）。"""
from vllm.model_executor.layers.quantization.compressed_tensors.utils import should_ignore_layer
# 与 convert_glm53_ct_int4.py 的 ignore 表同步（含 *_SHARED_EXPERTS_INT4=False 分支）
IGN = ["lm_head", "head", "embed", "*norm*", "*mlp.gate",
       "*self_attn.indexer.weights_proj*", "*self_attn.indexer.wk*",
       "*shared_expert_gate*", "*mlp.shared_experts.*", "*eh_proj"]
CASES = [
    ("model.layers.0.self_attn.indexer.wk_weights_proj", True,  "融合模块->未量化"),
    ("model.layers.0.self_attn.indexer.wk",              True,  "wk->未量化"),
    ("model.layers.0.self_attn.indexer.weights_proj",    True,  "weights_proj->未量化"),
    ("model.layers.0.self_attn.indexer.wq_b",            False, "wq_b->量化int4"),
    ("model.layers.0.self_attn.indexer.k_norm",          True,  "k_norm->未量化"),
    ("model.layers.0.mlp.gate_proj",                     False, "dense MLP->量化"),
    ("model.layers.0.mlp.gate",                          True,  "路由器->未量化"),
    ("model.layers.0.mlp.shared_experts.gate_proj",      True,  "共享专家->未量化"),
    ("model.layers.78.eh_proj",                          True,  "eh_proj->未量化"),
    ("model.layers.5.mlp.experts.3.up_proj",             False, "专家->量化"),
    ("model.layers.0.self_attn.q_b_proj",                False, "注意力->量化"),
    ("model.layers.0.self_attn.kv_a_proj_with_mqa",      False, "kv_a->量化"),
]
bad = 0
for name, want, tag in CASES:
    got = bool(should_ignore_layer(name, ignore=IGN, use_fnmatch=True))
    ok = (got == want); bad += (not ok)
    print("  %-52s ignore=%-5s %s  (%s)" % (name, got, "OK" if ok else "MISMATCH", tag))
print("不一致:", bad)
