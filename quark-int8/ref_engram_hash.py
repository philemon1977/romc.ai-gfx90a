#!/usr/bin/env python3
"""离线复算 engram 的 n-gram 哈希行号，与运行时 dump 的 hash_ids 逐位比对。

规格 = checkpoint 自带 inference/engram.py（EngramLayout + NgramHashState），
完全按原文实现（不是从 vLLM 抄）：
  primes[layer][ngram-1][head]：从 engram_vocab_size-1 起找素数，**全局去重、按序发放**
  multipliers[layer][shift]：np.random.default_rng(10007*layer_id).integers(0,bound,4)*2+1
  offsets[layer]：对 (3 ngram × 8 head) 的素数序列做 cumsum([0, *sizes[:-1]])
  hash(pos) = XOR_{shift<ngram} (tokens[pos-shift] * mult[shift]) % primes[ngram-1] + offsets
  其中 tokens 经 compressed token map（归一化后同形的 token 合并），越界/开头/DEAD 用 pad_id
"""
import json, sys
import numpy as np
import torch

CKPT = "/models"
def main():
    d = torch.load("/tmp/dsv41_hash_ids.pt", map_location="cpu", weights_only=False)
    got = d["hash_ids"].to(torch.int64)
    print(f"运行时 hash_ids{tuple(got.shape)} rank窗口=[{d['vocab_start']},{d['vocab_start']+d['part_rows']}) 表总行={d['rows_total']}")
    ids = torch.load("/tmp/dsv41_L0_in.pt", map_location="cpu", weights_only=False)["layer0_input_ids"]
    assert ids is not None, "dump 里没有 input_ids"
    ids = ids.reshape(-1).tolist()
    print(f"prompt token ids ({len(ids)}) = {ids}")

    cfg = json.load(open(f"{CKPT}/config.json"))["text_config"]
    layer_ids = tuple(cfg["engram_layer_ids"]); mn = cfg["engram_max_ngram_size"]
    nh = cfg["engram_n_heads"]; vbase = cfg["engram_vocab_size"]
    pad_tok = cfg["engram_pad_token_id"]; cvs = cfg["engram_compressed_vocab_size"]
    print(f"layer_ids={layer_ids} max_ngram={mn} n_heads={nh} vocab={vbase} pad_id={pad_tok} compressed={cvs}")

    # ---- 压缩 token map（用 tokenizer 复刻参考的归一化规则）----
    from tokenizers import Regex, normalizers
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT)
    sentinel = "\ue000"
    norm = normalizers.Sequence([normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(),
                                 normalizers.Lowercase(), normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                                 normalizers.Replace(Regex(r"^ $"), sentinel), normalizers.Strip(),
                                 normalizers.Replace(sentinel, " ")])
    backend = tok.backend_tokenizer
    k2n, lookup = {}, [0]*len(tok)
    for tid in range(len(tok)):
        txt = backend.decode([tid], skip_special_tokens=False)
        if "\ufffd" in txt: key = backend.id_to_token(tid)
        else:
            nz = norm.normalize_str(txt); key = nz if nz else txt
        if key not in k2n: k2n[key] = len(k2n)
        lookup[tid] = k2n[key]
    print(f"复算压缩词表大小 = {len(k2n)}（配置要求 {cvs}）{'✅' if len(k2n)==cvs else '❌'}")

    # ---- 素数 / offsets / multipliers（严格按参考顺序）----
    def is_prime(n):
        if n < 2: return False
        if n % 2 == 0: return n == 2
        i = 3
        while i*i <= n:
            if n % i == 0: return False
            i += 2
        return True
    def next_prime(start, seen):
        c = start + 1
        while not is_prime(c) or c in seen: c += 1
        return c
    primes, seen = [], set()
    for _ in layer_ids:
        per = []
        for _ in range(mn-1):
            cur = vbase - 1; sizes = []
            for _ in range(nh):
                cur = next_prime(cur, seen); seen.add(cur); sizes.append(cur)
            per.append(tuple(sizes))
        primes.append(tuple(per))
    flat = [[p for per in layer for p in per] for layer in primes]
    offsets = [np.cumsum([0, *s[:-1]]) for s in flat]
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // cvs) // 2)
    mults = []
    for lid in layer_ids:
        g = np.random.default_rng(10007*lid)
        mults.append(torch.tensor(g.integers(0, bound, size=(mn,), dtype=np.int64)*2+1))
    offsets = torch.tensor(np.array(offsets))

    # ---- 逐层算 hash（只比对 dump 的那一层：layer 1 → index 0）----
    pad_id = lookup[pad_tok]
    comp = torch.tensor([lookup[t] for t in ids], dtype=torch.int64)
    T = comp.shape[0]
    positions = torch.arange(T)
    blocked = torch.zeros(T, dtype=torch.bool)
    toks = []
    for shift in range(mn):
        src = comp[(positions - shift).clamp_min(0)]
        blocked = blocked | (positions < shift)
        toks.append(torch.where(blocked, torch.tensor(pad_id), src))
    tokens = torch.stack(toks, -1).unsqueeze(0)          # [1,T,mn]
    products = tokens.unsqueeze(2) * torch.stack(mults).view(len(layer_ids), mn)[None, None, :, :]
    rolling, hashes = products[..., 0], []
    for i in range(1, mn):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % torch.tensor(np.array(primes))[:, i-1])
    mine = (torch.cat(hashes, dim=-1) + offsets)
    print(f"复算 hash_ids{tuple(mine.shape)}")
    # dump 的 hash_ids 来自某一层；逐层比一遍，报告哪层匹配
    for li, lid in enumerate(layer_ids):
        ref = mine[0, :, li, :]                              # [T, 24]
        if ref.shape != got.shape:
            print(f"  layer {lid}: 形状不符 {tuple(ref.shape)} vs {tuple(got.shape)}"); continue
        same = bool(torch.equal(ref, got))
        md = int((ref - got).abs().max())
        print(f"  layer {lid}: 逐位相同={same} 最大行号差={md}  {'✅' if same else '❌'}")
    print(f"  复算 min={int(mine.min())} max={int(mine.max())}  运行时 min={int(got.min())} max={int(got.max())}")
main()
