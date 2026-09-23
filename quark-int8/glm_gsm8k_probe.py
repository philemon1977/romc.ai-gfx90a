#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GSM8K 6 题贪婪命中率探针（独立文件，避免 heredoc 变量展开问题）。用法: python3 glm_gsm8k_probe.py <port> <model>"""
import json, re, sys, urllib.request
PORT = sys.argv[1] if len(sys.argv) > 1 else "8121"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "glm-5.3"
Q = [
    ("Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "10"),
    ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", "624"),
    ("Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. In the morning she gives 15 cups, in the afternoon 25 cups, for a flock of 20 chickens. How many cups in the final meal?", "20"),
    ("Kylar buys 16 glasses at $5 each, but every second glass costs 60% of the price. How much does he pay?", "64"),
    ("Betty needs $100. She has half. Parents give $15, grandparents twice that. How much more does she need?", "35"),
]
ok = 0
for q, gold in Q:
    body = json.dumps({"model": MODEL, "prompt": "Question: " + q + "\nAnswer:",
                       "max_tokens": 256, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:" + PORT + "/v1/completions", body,
                                 {"Content-Type": "application/json"})
    txt = json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]["text"]
    hit = bool(re.search(r"(?<![0-9.])" + gold + r"(?![0-9])", txt))
    ok += hit
    print("  %s gold=%-4s -> %r" % ("OK" if hit else "XX", gold, txt[:130]))
print("GSM8K 命中 %d/6" % ok)
