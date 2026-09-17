#!/usr/bin/env python3
"""Pin down vLLM's WNA16 int4 B layout + nibble convention by a minimal experiment:
M=1 token, TOPK=1 expert => the MoE reduces to one GEMV, so the incumbent kernel's output
for that single (token, expert) pair can be compared against an explicit dequant reference
built from a candidate packing assumption. Whichever assumption matches IS the convention.

Candidate assumptions for B[E, N, K/8] int32:
  A) 8 nibbles per word, low-nibble-first:  w[8i+j] = (word >> 4j) & 0xF
  B) 8 nibbles per word, high-nibble-first: w[8i+j] = (word >> 4(7-j)) & 0xF
  C) 2 int4 per element (as the kernel's (offs_k//2) indexing suggests), low-first:
       w[2i+j] = (word >> 4j) & 0xF, tensor last dim = K/2
"""
import torch

from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_wna16_triton_kernel,
)

E, HIDDEN, GROUP = 8, 256, 128          # tiny: 1 expert used, K=256, N=64
NPER, TOPK, M = 64, 1, 1
CFG = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
       "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": 4, "num_stages": 2}
dev = "cuda"


def unpack(words, K, mode):
    """words: [N, KDIM] int32 -> [N, K] float (int4 values only, no scale)."""
    if mode == "2per":
        assert words.shape[1] * 2 == K
        out = torch.empty(words.shape[0], K, dtype=torch.float32, device=dev)
        for j in range(2):
            v = (words.to(torch.int64) >> (4 * j)) & 0xF
            v = torch.where(v >= 8, v - 16, v).float()
            out[:, j::2] = v
        return out
    assert words.shape[1] * 8 == K, (words.shape, K)
    out = torch.empty(words.shape[0], K, dtype=torch.float32, device=dev)
    for j in range(8):
        sh = 4 * j if mode == "lowfirst" else 4 * (7 - j)
        v = (words.to(torch.int64) >> sh) & 0xF
        v = torch.where(v >= 8, v - 16, v).float()
        out[:, j::8] = v
    return out


def run_incumbent(B, S, x, kdim):
    M_, K = x.shape
    c = torch.zeros(M_, TOPK, NPER, dtype=torch.bfloat16, device=dev)
    sid = torch.zeros(CFG["BLOCK_SIZE_M"], dtype=torch.int32, device=dev)   # token 0
    eid = torch.zeros(1, dtype=torch.int32, device=dev)                     # expert 0
    npp = torch.tensor([CFG["BLOCK_SIZE_M"]], dtype=torch.int32, device=dev)
    tw = torch.ones(M_, TOPK, dtype=torch.float32, device=dev)
    invoke_fused_moe_wna16_triton_kernel(
        x, B, c, S, None, tw, sid, eid, npp,
        False, TOPK, CFG, __import__("triton.language", fromlist=["x"]).bfloat16,
        False, True, [0, GROUP])
    return c[0, 0].float()


def main():
    torch.manual_seed(0)
    x = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device=dev)
    S = (torch.rand(E, NPER, HIDDEN // GROUP, dtype=torch.float32, device=dev) + 0.5).to(torch.bfloat16)

    # 参考：先造"逻辑 int4 权重"，再按各假设打包
    w_log = torch.randint(-8, 8, (E, NPER, HIDDEN), dtype=torch.float32, device=dev)
    scal = S.float().repeat_interleave(GROUP, dim=2)
    ref_log = (w_log[0] * scal[0]) @ x[0].float()      # 逻辑参考（与打包无关）

    print("逻辑参考 |out| 均值 = %.4f" % ref_log.abs().mean().item())
    for mode, kdim in (("lowfirst", HIDDEN // 8), ("highfirst", HIDDEN // 8), ("2per", HIDDEN // 2)):
        words = torch.zeros(E, NPER, kdim, dtype=torch.int32, device=dev)
        wi = (w_log.to(torch.int64) + 8) & 0xF   # offset-binary: 真值 = nibble - 8
        if mode == "2per":
            for j in range(2):
                words |= (wi[:, :, j::2] << (4 * j)).to(torch.int32)
        else:
            for j in range(8):
                sh = 4 * j if mode == "lowfirst" else 4 * (7 - j)
                words |= (wi[:, :, j::8] << sh).to(torch.int32)
        got = run_incumbent(words, S, x, kdim)
        denom = ref_log.abs().mean().item() + 1e-6
        rel = (got - ref_log).abs().mean().item() / denom
        tag = "✅ 匹配" if rel < 0.05 else "❌ 不匹配"
        print(f"  假设 {mode:10s} (last dim = K/{HIDDEN//kdim})  输出|均值|={got.abs().mean().item():8.4f}  "
              f"相对误差={rel*100:7.2f}%  {tag}")


if __name__ == "__main__":
    main()
