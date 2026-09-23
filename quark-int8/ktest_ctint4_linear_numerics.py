#!/usr/bin/env python3
"""CT-int4 运行期内核 vs "反量化 + matmul" 参考 —— 用**真权重**对拍。

为什么要它：本轮已用四个往返测试证明 KV/注意力各核正确，但**没人验证过
`CompressedTensorsWNA16` + `TritonW4A16LinearKernel` 在真权重上算出来的数**。
而 09-12 准入调研 §10.5 明确警告过这类配置"会**静默算错**（比报错贵得多）"。
这个脚本把 checkpoint 的真 `weight_packed`/`weight_scale` 灌进 vLLM 真实
量化线性层，跑一次 forward，与"按 CT 约定反量化再 bf16 matmul"逐元素比。

靶子默认取 `layers.0.ffn.shared_experts.w1`（out=2304, in=5120, g=32, 干净 CT 线性）。

用法（容器内，需 GPU）:
  python3 ktest_ctint4_linear_numerics.py [tensor_prefix ...]
"""
import json
import os
import sys

import torch

MODEL = "/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4"

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29581")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

hf = json.load(open(os.path.join(MODEL, "config.json")))
tc = dict(hf["text_config"])
tc.update(num_hidden_layers=2, n_routed_experts=8, num_nextn_predict_layers=0,
          engram_layer_ids=[], index_source_layer_ids=[], kv_source_layer_ids=[],
          compress_ratios=[0, 1], num_hash_layers=0)

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.transformers_utils.config import _register_config_class, _CONFIG_REGISTRY  # noqa: E402

_register_config_class("deepseek_v41", _CONFIG_REGISTRY["deepseek_v41"])

model_config = ModelConfig(
    model=MODEL, tokenizer=MODEL, tokenizer_mode="deepseek_v41",
    trust_remote_code=True, dtype="bfloat16", max_model_len=512,
    enforce_eager=True, hf_overrides=tc,
)
vllm_config = VllmConfig(model_config=model_config)
vllm_config.model_config.dtype = torch.bfloat16

from vllm.distributed import (  # noqa: E402
    init_distributed_environment, initialize_model_parallel,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear  # noqa: E402
from safetensors import safe_open  # noqa: E402

WANT = sys.argv[1:] or ["layers.0.ffn.shared_experts.w1"]
owm = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]


def ckpt(name):
    with safe_open(os.path.join(MODEL, owm[name]), "pt") as f:
        return f.get_tensor(name)


def dequant_ref(wp, ws, group=32):
    """CT/**uint4b8** 约定：weight_packed int32 [out, in/8]，低 nibble 在前；
    code(0..15) 是**无符号偏移码**，真值 = (code - 8) * scale（内核 `zp_bias=8`）。

    ⚠️ 这里踩过一次坑：最初按"两补码有符号 (-8..7)"解码，得到 relERR 133%、cos≈-0.55，
    差点误判成"运行期算错"。用**源 FP8 权重**当裁判才定案：
      两补码口径 cos=-0.554 / relERR=337%   ✗
      无符号-8 口径 cos=+0.995 / relERR=10.3% ✓（= 4bit 量化噪声）
    """
    shifts = torch.arange(8, device=wp.device, dtype=torch.int32) * 4
    nib = ((wp.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF).reshape(wp.shape[0], -1)
    codes = (nib - 8).to(torch.float32)
    assert codes.shape[1] == ws.shape[1] * group, (codes.shape, ws.shape)
    return codes * ws.float().repeat_interleave(group, dim=1)


init_distributed_environment(world_size=1, rank=0, local_rank=0,
                            distributed_init_method="env://", backend="gloo")
qc = vllm_config.quant_config
print("quant_config:", type(qc).__name__)

bad = 0
with set_current_vllm_config(vllm_config):
    initialize_model_parallel(tensor_model_parallel_size=1)
    for base in WANT:
        wp = ckpt(base + ".weight_packed")
        ws = ckpt(base + ".weight_scale")
        shape = ckpt(base + ".weight_shape").tolist()
        out_f, in_f = int(shape[0]), int(shape[1])
        w_ref = dequant_ref(wp, ws)                       # [out, in]
        print(f"\n=== {base}: packed{tuple(wp.shape)} scale{tuple(ws.shape)} "
              f"weight_shape={shape} -> ref{tuple(w_ref.shape)}")
        layer = ColumnParallelLinear(
            input_size=in_f, output_size=out_f, bias=False, quant_config=qc,
            prefix=base, disable_tp=True, params_dtype=torch.bfloat16,
            return_bias=False)
        prm = dict(layer.named_parameters())
        # TP1 ⇒ 形状与 checkpoint 完全一致，直接拷（同时也验证 CT 参数量纲）
        assert prm["weight_packed"].shape == wp.shape, (prm["weight_packed"].shape, wp.shape)
        assert prm["weight_scale"].shape == ws.shape, (prm["weight_scale"].shape, ws.shape)
        prm["weight_packed"].data.copy_(wp)
        prm["weight_scale"].data.copy_(ws)
        layer = layer.to("cuda")
        # ★ 引擎在装载后会调这一步：CT kernel 在其中 repack_w_q（[out,in/8] → 内核要的
        #   [in,out/8] GPTQ 顺序）。漏掉它 ⇒ 内核断言 b_q.shape==(K,N//8) 直接炸。
        layer.quant_method.process_weights_after_loading(layer)
        prm = dict(layer.named_parameters())
        print(f"    after repack: weight_packed{tuple(prm['weight_packed'].shape)} "
              f"weight_scale{tuple(prm['weight_scale'].shape)}")
        x = torch.randn(8, in_f, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            y = layer(x)
        if isinstance(y, tuple):
            y = y[0]
        with torch.no_grad():
            y_ref = torch.nn.functional.linear(x, w_ref.to("cuda").to(torch.bfloat16))
        d = (y.float() - y_ref.float())
        denom = y_ref.float().abs().mean().clamp_min(1e-6)
        rel = (d.abs().mean() / denom).item()
        cos = torch.nn.functional.cosine_similarity(
            y.float().flatten(), y_ref.float().flatten(), dim=0).item()
        ok = rel < 0.02 and cos > 0.999
        bad += 0 if ok else 1
        print(f"    runtime vs dequant-ref: relERR={rel*100:.3f}%  cos={cos:.6f}  "
              f"max|Δ|={d.abs().max().item():.5g}  mean|y|={y_ref.float().abs().mean().item():.5g}"
              f"  -> {'PASS' if ok else 'FAIL'}")

print("\n[PASS] CT-int4 运行期内核与反量化参考一致（线性层）" if bad == 0
      else f"\n[FAIL] {bad} 个张量不符 —— CT-int4 运行期路径在真权重上算错")
raise SystemExit(1 if bad else 0)
