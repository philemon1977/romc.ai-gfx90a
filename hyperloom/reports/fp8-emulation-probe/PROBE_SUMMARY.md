# FP8->BF16 dequant-emulation probe (MI250X, TP8, vLLM)

- measured_at: 2026-09-15T19:26:27Z
- model: /mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-FP8
- prompt tokens: 776

## Single stream (median of 3)
- TTFT: 377 ms
- decode: 23.4 tok/s
- TPOT: 43 ms

## Concurrency
- conc 8 (osl 128): aggregate 60.8 output tok/s, mean TTFT 5314 ms, mean TPOT 91 ms, wall 16.9 s
- conc 32 (osl 128): aggregate 236.7 output tok/s, mean TTFT 2793 ms, mean TPOT 113 ms, wall 17.3 s

Reference on the same host: llama.cpp Q8_0 arm = 47.28 tok/s (decode).
