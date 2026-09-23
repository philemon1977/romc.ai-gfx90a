# -*- coding: utf-8 -*-
"""1M 上下文可行性算术 + agent 会话压测（多轮、前缀缓存、长上下文 TTFT）。

KV 每 token 每层的构成（GLM-5.3, MLA + DSA）：
  MLA: kv_lora_rank 512 + qk_rope_head_dim 64 = 576 值/token/层（fp8_ds_mla 缓存 ≈ 576 B + scale）
  DSA indexer: index_head_dim 128 + 4 B scale = 132 B/token/层
"""
import json, sys

CFG = json.load(open("/mnt/kioxia-cm6-3t8/ai/models/GLM/GLM-5.3-753B/FP8/config.json"))
L = CFG["num_hidden_layers"]
kv = CFG["kv_lora_rank"] + CFG["qk_rope_head_dim"]          # 576
idx = CFG["index_head_dim"] + 4                              # 132
per_tok = (kv + 20 + idx) * L                                # 每 token 全模型字节数
print("层数=%d  MLA=%dB  indexer=%dB  => 每 token ≈ %.1f KB" % (L, kv + 20, idx, per_tok / 1024))
for n in (32768, 262144, 1048576):
    print("   上下文 %9d token -> KV %.2f GiB" % (n, per_tok * n / 2**30))
print()
print("预算：权重 401.6 GiB（int4） + KV 预算 ≈ %.0f GiB（util 0.97）" % (496 - 401.6))
print("⇒ 单条 1M 上下文需 %.1f GiB，余量 %.1f GiB" % (per_tok * 1048576 / 2**30, (496 - 401.6) - per_tok * 1048576 / 2**30))
print("⇒ 并发 N 条 1M：N=%.1f 条（按 KV 预算）" % ((496 - 401.6) * 2**30 / (per_tok * 1048576)))
