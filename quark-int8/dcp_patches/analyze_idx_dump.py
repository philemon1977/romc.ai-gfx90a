#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判定针尖 MISS 的归因：针尖是否落在 DSA top-2048 选择集里（选择 vs 注意力）。

背景（2026-09-20 评审）：GLM-5.3 config index_topk=2048 ⇒ 9.2K 上下文里每条 query
只对 2048 个被选中的 token 做注意力。针尖 MISS 有两种完全不同的成因：
  (a) 针尖没进 top-2048（DSA 选择性质，与我们的 Triton 内核/DCP 无关）；
  (b) 针尖进了选择集但输出错（这才指向注意力内核/后续 DCP 合并）。
不区分这两者，⓪ 门的 HIT/MISS 无法归因。

用法（**在服务容器内跑**，dump 在容器 /tmp，仓库未挂进容器）：
  1) 起服时带 -e DSV41_IDX_DUMP=1（我们 ops 1475-1503 会把每层 logits/topk_indices
     dump 到 /tmp/idx_dump/layer_*.pt，按层去重）。
  2) 用 agent_bench2.py --save-prompt /tmp/p.txt 落盘最后一次针尖 prompt。
  3) docker exec -i glm53-int4 python3 - < analyze_idx_dump.py --prompt /tmp/p.txt \
       --tokenizer /models --code 427109 --dump-dir /tmp/idx_dump
"""
import argparse, glob, os, re


def needle_token_span(tokenizer, prompt, code):
    """返回 code 在 prompt 里的 token 下标区间（含端点）。"""
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    cids = tokenizer(" " + code, add_special_tokens=False).input_ids
    if not cids:
        return None
    n = len(cids)
    for i in range(len(ids) - n + 1):
        if ids[i:i + n] == cids:
            return (i, i + n - 1)
    # 退化：只用数字串的前 4 位再找一次（避免 tokenizer 把 6 位数字切法不同）
    for k in (4, 3, 2):
        cids = tokenizer(" " + code[:k], add_special_tokens=False).input_ids
        for i in range(len(ids) - len(cids) + 1):
            if ids[i:i + len(cids)] == cids:
                return (i, i + len(cids) - 1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="/tmp/idx_dump")
    ap.add_argument("--prompt", required=True, help="agent_bench2 --save-prompt 落的文件")
    ap.add_argument("--tokenizer", default="/models", help="容器内模型目录（含 tokenizer）")
    ap.add_argument("--code", default="427109")
    ap.add_argument("--window", type=int, default=4, help="针尖位置两侧的容差 token 数")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    prompt = open(a.prompt, encoding="utf8").read()
    span = needle_token_span(tok, prompt, a.code)
    total = len(tok(prompt, add_special_tokens=False).input_ids)
    if span is None:
        print("[FATAL] 在 prompt 里定位不到针尖 token —— 换 --code 或检查 save-prompt")
        return
    print("prompt tokens=%d 针尖 token 区间=[%d,%d]" % (total, span[0], span[1]))

    files = sorted(glob.glob(os.path.join(a.dump_dir, "layer_*.pt")))
    print("dump 层数=%d" % len(files))
    if not files:
        print("[提示] 没 dump ⇒ 起服时没设 DSV41_IDX_DUMP=1")
        return

    hits = misses = unknown = 0
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        topk = d.get("topk_indices")
        ks = d.get("cu_ks"); ke = d.get("cu_ke")
        if topk is None or ks is None:
            unknown += 1; continue
        row = int(topk.shape[0]) - 1          # 最后一行 = 问题末尾 token（最靠近针尖）
        sel = topk[row].tolist()
        row_start = int(ks[row].item()); row_end = int(ke[row].item())
        # 针尖 seq 位置 → gathered buffer 偏移（chunk 覆盖 0..end 时偏移即 seq 位置）
        buf_pos = span[0]
        lo, hi = buf_pos - a.window, span[1] + a.window
        valid = [s for s in sel if s is not None and s >= 0]
        inset = any(lo <= s <= hi for s in valid)
        hits += int(inset); misses += int(not inset)
        print("  %-22s row=%d ctx=[%d,%d) topk=%d valid=%d 选中位置 min/max=%s/%s 针尖在集合内=%s"
              % (os.path.basename(f), row, row_start, row_end, len(sel), len(valid),
                 min(valid) if valid else "-", max(valid) if valid else "-", inset))
    print("=== 归因：针尖被选中的层 %d，未被选中的层 %d，无法判定 %d ===" % (hits, misses, unknown))
    if misses > hits:
        print("⇒ 主因是 DSA top-k 选择（不是注意力内核/DCP）：本轮 HIT/MISS 不能作为稀疏路径判据，",
              "   改用 DCP parity 判据（同一 prompt 下 DCP=1 vs DCP=N 输出一致性）。")
    else:
        print("⇒ 针尖确实进了选择集却答错：此时才应怀疑注意力路径（跑 dcp_patches/test_lse_merge.py 的尺子）")


if __name__ == "__main__":
    main()
